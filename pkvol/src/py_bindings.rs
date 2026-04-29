//! PyO3 bindings — only built with the `python` feature.

use numpy::{IntoPyArray, PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::angular_search::AngularTree;
use crate::ecdf2d::EcdfBackend;
use crate::pk_aggregate::{lambda_per_query, measure_pk, AngularVar, MeasureConfig, ShellSpec};

fn parse_angular_variable(s: &str) -> PyResult<AngularVar> {
    match s {
        "theta" => Ok(AngularVar::Theta),
        "theta2" => Ok(AngularVar::Theta2),
        "omega" => Ok(AngularVar::Omega),
        other => Err(PyValueError::new_err(format!(
            "angular_variable must be one of 'theta', 'theta2', 'omega'; got '{other}'"
        ))),
    }
}

fn parse_backend(s: &str) -> PyResult<EcdfBackend> {
    match s {
        "ecdf" | "sweep" => Ok(EcdfBackend::Sweep),
        "histogram" => Ok(EcdfBackend::Histogram),
        other => Err(PyValueError::new_err(format!(
            "backend must be 'ecdf' or 'histogram'; got '{other}'"
        ))),
    }
}

fn parse_intervals(arr: PyReadonlyArray2<i64>, n_z: usize) -> PyResult<Vec<(usize, usize)>> {
    let view = arr.as_array();
    let shape = view.shape();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(PyValueError::new_err(
            "z_intervals must have shape (n_intervals, 2)".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(shape[0]);
    for row in view.rows() {
        let l = row[0];
        let r = row[1];
        let li = if l < 0 { usize::MAX } else { l as usize };
        let ri = if r < 0 || (r as usize) >= n_z {
            return Err(PyValueError::new_err(format!(
                "z_intervals row r={r} out of range (n_z={n_z})"
            )));
        } else {
            r as usize
        };
        if li != usize::MAX && li >= ri {
            return Err(PyValueError::new_err(format!(
                "z_intervals row l={l} >= r={r} (must satisfy l < r)"
            )));
        }
        out.push((li, ri));
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (ra_gal, dec_gal, z_gal, weights_gal, ra_query, dec_query,
                    theta_edges, z_edges, k_values, theta_max,
                    z_intervals=None, query_weights=None,
                    exclude_self=false, self_match_tol=1e-9,
                    angular_variable="theta", backend="ecdf",
                    n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn measure_pk_py<'py>(
    py: Python<'py>,
    ra_gal: PyReadonlyArray1<f64>,
    dec_gal: PyReadonlyArray1<f64>,
    z_gal: PyReadonlyArray1<f64>,
    weights_gal: Option<PyReadonlyArray1<f64>>,
    ra_query: PyReadonlyArray1<f64>,
    dec_query: PyReadonlyArray1<f64>,
    theta_edges: PyReadonlyArray1<f64>,
    z_edges: PyReadonlyArray1<f64>,
    k_values: PyReadonlyArray1<f64>,
    theta_max: f64,
    z_intervals: Option<PyReadonlyArray2<i64>>,
    query_weights: Option<PyReadonlyArray1<f64>>,
    exclude_self: bool,
    self_match_tol: f64,
    angular_variable: &str,
    backend: &str,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let av = parse_angular_variable(angular_variable)?;
    let be = parse_backend(backend)?;

    let ra = ra_gal.as_slice()?;
    let dec = dec_gal.as_slice()?;
    let z = z_gal.as_slice()?;
    if ra.len() != dec.len() || ra.len() != z.len() {
        return Err(PyValueError::new_err(
            "ra_gal, dec_gal, z_gal must have equal length",
        ));
    }
    let w_owned: Vec<f64>;
    let w: &[f64] = match &weights_gal {
        Some(arr) => {
            let s = arr.as_slice()?;
            if s.len() != ra.len() {
                return Err(PyValueError::new_err(
                    "weights_gal length must equal ra_gal length",
                ));
            }
            s
        }
        None => {
            w_owned = vec![1.0; ra.len()];
            &w_owned
        }
    };

    let raq = ra_query.as_slice()?;
    let decq = dec_query.as_slice()?;
    if raq.len() != decq.len() {
        return Err(PyValueError::new_err(
            "ra_query and dec_query must have equal length",
        ));
    }

    let qw_buf: Option<Vec<f64>>;
    let qw: Option<&[f64]> = match &query_weights {
        Some(arr) => {
            let s = arr.as_slice()?;
            if s.len() != raq.len() {
                return Err(PyValueError::new_err(
                    "query_weights length must equal ra_query length",
                ));
            }
            qw_buf = None;
            Some(s)
        }
        None => {
            qw_buf = None;
            None
        }
    };
    let _ = qw_buf;

    let theta_edges_v = theta_edges.as_slice()?.to_vec();
    let z_edges_v = z_edges.as_slice()?.to_vec();
    let k_values_v = k_values.as_slice()?.to_vec();

    if theta_edges_v.is_empty() {
        return Err(PyValueError::new_err("theta_edges must be non-empty"));
    }
    if z_edges_v.is_empty() {
        return Err(PyValueError::new_err("z_edges must be non-empty"));
    }
    if k_values_v.is_empty() {
        return Err(PyValueError::new_err("k_values must be non-empty"));
    }

    // Translate theta_edges into the chosen x variable.
    let u_edges: Vec<f64> = theta_edges_v
        .iter()
        .map(|&t| match av {
            AngularVar::Theta => t,
            AngularVar::Theta2 => t * t,
            AngularVar::Omega => 1.0 - t.cos(),
        })
        .collect();

    // Intervals.
    let intervals = if let Some(arr) = z_intervals {
        parse_intervals(arr, z_edges_v.len())?
    } else {
        crate::shell_counts::adjacent_intervals(z_edges_v.len())
    };
    if intervals.is_empty() {
        return Err(PyValueError::new_err(
            "no shell intervals (need len(z_edges) >= 2 or explicit z_intervals)",
        ));
    }

    let cfg = MeasureConfig {
        theta_max,
        angular_variable: av,
        backend: be,
        exclude_self,
        self_match_tol,
    };
    let shell = ShellSpec {
        z_edges: z_edges_v.clone(),
        intervals: intervals.clone(),
    };

    // Build tree (release the GIL for the heavy work).
    let pool = if let Some(nt) = n_threads {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .map_err(|e| PyValueError::new_err(format!("rayon pool: {e}")))?,
        )
    } else {
        None
    };

    let result = py.allow_threads(|| {
        let tree = AngularTree::new(ra, dec);
        let run = || measure_pk(&tree, z, w, raq, decq, qw, &u_edges, &shell, &k_values_v, &cfg);
        if let Some(p) = pool.as_ref() {
            p.install(run)
        } else {
            run()
        }
    });

    let n_k = result.n_k;
    let n_u = result.n_u;
    let n_int = result.n_intervals;

    let f_arr = PyArray3::<f64>::from_vec3_bound(py, &vec_to_3d(result.f, n_k, n_u, n_int))
        .map_err(|e| PyValueError::new_err(format!("from_vec3_bound: {e:?}")))?;
    let p_arr = PyArray3::<f64>::from_vec3_bound(py, &vec_to_3d(result.p, n_k, n_u, n_int))
        .map_err(|e| PyValueError::new_err(format!("from_vec3_bound: {e:?}")))?;

    let dict = PyDict::new_bound(py);
    dict.set_item("F", f_arr)?;
    dict.set_item("P", p_arr)?;
    dict.set_item("k_values", k_values_v.clone().into_pyarray_bound(py))?;
    dict.set_item("theta_edges", theta_edges_v.clone().into_pyarray_bound(py))?;
    dict.set_item("z_edges", z_edges_v.clone().into_pyarray_bound(py))?;
    let intervals_flat: Vec<i64> = intervals
        .iter()
        .flat_map(|&(l, r)| {
            let li = if l == usize::MAX { -1 } else { l as i64 };
            [li, r as i64]
        })
        .collect();
    let intervals_arr = PyArray1::from_vec_bound(py, intervals_flat).reshape((intervals.len(), 2))?;
    dict.set_item("z_intervals", intervals_arr)?;
    dict.set_item("total_weight", result.total_weight)?;
    dict.set_item("n_query_used", result.n_query_used)?;
    dict.set_item("mean_candidates", result.mean_candidates)?;
    dict.set_item("mean_total_count", result.mean_total_count)?;
    dict.set_item("query_valid_fraction", 1.0_f64)?;
    dict.set_item("angular_variable", angular_variable.to_string())?;
    dict.set_item("backend", backend.to_string())?;

    Ok(dict)
}

fn vec_to_3d(v: Vec<f64>, n_k: usize, n_u: usize, n_int: usize) -> Vec<Vec<Vec<f64>>> {
    let mut out = Vec::with_capacity(n_k);
    let mut idx = 0;
    for _ in 0..n_k {
        let mut plane = Vec::with_capacity(n_u);
        for _ in 0..n_u {
            let mut row = Vec::with_capacity(n_int);
            for _ in 0..n_int {
                row.push(v[idx]);
                idx += 1;
            }
            plane.push(row);
        }
        out.push(plane);
    }
    out
}

#[pyfunction]
#[pyo3(signature = (ra_gal, dec_gal, z_gal, weights_gal, ra_query, dec_query,
                    theta_edges, z_edges, theta_max, z_intervals=None,
                    angular_variable="theta", backend="ecdf",
                    n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn lambda_per_query_py<'py>(
    py: Python<'py>,
    ra_gal: PyReadonlyArray1<f64>,
    dec_gal: PyReadonlyArray1<f64>,
    z_gal: PyReadonlyArray1<f64>,
    weights_gal: Option<PyReadonlyArray1<f64>>,
    ra_query: PyReadonlyArray1<f64>,
    dec_query: PyReadonlyArray1<f64>,
    theta_edges: PyReadonlyArray1<f64>,
    z_edges: PyReadonlyArray1<f64>,
    theta_max: f64,
    z_intervals: Option<PyReadonlyArray2<i64>>,
    angular_variable: &str,
    backend: &str,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let av = parse_angular_variable(angular_variable)?;
    let be = parse_backend(backend)?;
    let ra = ra_gal.as_slice()?;
    let dec = dec_gal.as_slice()?;
    let z = z_gal.as_slice()?;
    let w_owned: Vec<f64>;
    let w: &[f64] = match &weights_gal {
        Some(arr) => arr.as_slice()?,
        None => {
            w_owned = vec![1.0; ra.len()];
            &w_owned
        }
    };
    let raq = ra_query.as_slice()?;
    let decq = dec_query.as_slice()?;
    let theta_edges_v = theta_edges.as_slice()?.to_vec();
    let z_edges_v = z_edges.as_slice()?.to_vec();
    let u_edges: Vec<f64> = theta_edges_v
        .iter()
        .map(|&t| match av {
            AngularVar::Theta => t,
            AngularVar::Theta2 => t * t,
            AngularVar::Omega => 1.0 - t.cos(),
        })
        .collect();

    let intervals = if let Some(arr) = z_intervals {
        parse_intervals(arr, z_edges_v.len())?
    } else {
        crate::shell_counts::adjacent_intervals(z_edges_v.len())
    };

    let cfg = MeasureConfig {
        theta_max,
        angular_variable: av,
        backend: be,
        exclude_self: false,
        self_match_tol: 1e-9,
    };
    let shell = ShellSpec {
        z_edges: z_edges_v.clone(),
        intervals: intervals.clone(),
    };

    let pool = if let Some(nt) = n_threads {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .map_err(|e| PyValueError::new_err(format!("rayon pool: {e}")))?,
        )
    } else {
        None
    };
    let n_q = raq.len();
    let n_u = u_edges.len();
    let n_int = intervals.len();
    let lambda = py.allow_threads(|| {
        let tree = AngularTree::new(ra, dec);
        let run = || lambda_per_query(&tree, z, w, raq, decq, &u_edges, &shell, &cfg);
        if let Some(p) = pool.as_ref() {
            p.install(run)
        } else {
            run()
        }
    });

    let arr = PyArray3::<f64>::from_vec3_bound(py, &vec_to_3d(lambda, n_q, n_u, n_int))
        .map_err(|e| PyValueError::new_err(format!("from_vec3_bound: {e:?}")))?;
    Ok(arr)
}

#[pyfunction]
fn haversine_py(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    crate::haversine::haversine(ra1, dec1, ra2, dec2)
}

#[pyfunction]
#[pyo3(signature = (ra, dec, ra_q, dec_q, theta_max))]
fn query_radius_py<'py>(
    py: Python<'py>,
    ra: PyReadonlyArray1<f64>,
    dec: PyReadonlyArray1<f64>,
    ra_q: f64,
    dec_q: f64,
    theta_max: f64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let ra_s = ra.as_slice()?;
    let dec_s = dec.as_slice()?;
    let tree = AngularTree::new(ra_s, dec_s);
    let mut buf = Vec::new();
    tree.query_radius(ra_q, dec_q, theta_max, &mut buf);
    let v: Vec<i64> = buf.into_iter().map(|i| i as i64).collect();
    Ok(PyArray1::from_vec_bound(py, v))
}

#[pyfunction]
#[pyo3(signature = (n, k, seed=0))]
fn subsample_indices_py<'py>(
    py: Python<'py>,
    n: usize,
    k: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let v = crate::randoms::subsample_indices(n, k, seed);
    let v: Vec<i64> = v.into_iter().map(|i| i as i64).collect();
    Ok(PyArray1::from_vec_bound(py, v))
}

#[pymodule]
fn _pkvol(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(measure_pk_py, m)?)?;
    m.add_function(wrap_pyfunction!(lambda_per_query_py, m)?)?;
    m.add_function(wrap_pyfunction!(haversine_py, m)?)?;
    m.add_function(wrap_pyfunction!(query_radius_py, m)?)?;
    m.add_function(wrap_pyfunction!(subsample_indices_py, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
