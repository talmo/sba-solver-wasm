# apex-solver-wasm - Task Tracking

## Project Status: COMPLETE

All core functionality is implemented and tested. The WASM module successfully runs bundle adjustment optimization in the browser.

## Completed Tasks

### 1. Investigation
- [x] Cloned apex-solver from https://github.com/amin-abouee/apex-solver
- [x] Documented API structure (Problem + LevenbergMarquardt pattern)
- [x] Identified WASM compatibility issues (memmap2, rayon, std::time)

### 2. Fork apex-solver for WASM
Location: `apex-solver-fork/`

**Changes made:**
- Feature flags: `io`, `parallel`, `cli`, `logging`, `visualization`
- Made `memmap2` and `rayon` optional dependencies
- Conditional compilation for parallel vs sequential iteration
- Replaced `std::time` with `web-time` crate for WASM timing
- Feature-gated `IoError` in error module
- Updated to Rust 2024 edition

### 3. Project Setup
- [x] Created `Cargo.toml` with WASM dependencies
- [x] Created `.cargo/config.toml` with SIMD128 and getrandom config
- [x] Configured `getrandom` with `wasm_js` backend

### 4. WASM Interface (`src/lib.rs`)
- [x] Data structures: `CameraParams`, `Observation`, `SolverConfig`, `BundleAdjustmentResult`
- [x] Custom `ReprojectionFactor` with analytical Jacobians
- [x] `WasmBundleAdjuster` class with JSON-based API
- [x] Huber and Cauchy robust loss support
- [x] SE3 manifold optimization for camera poses

### 5. Build & Test
- [x] Native compilation passing (`cargo check`)
- [x] Rust unit tests passing (`cargo test` - 4 tests)
- [x] WASM compilation passing (`cargo check --target wasm32-unknown-unknown`)
- [x] WASM build successful (`wasm-pack build --target web --release`)
- [x] Output: ~720KB WASM binary

### 6. Browser Demo
- [x] Created `examples/index.html` with interactive UI
- [x] Synthetic data generation for testing
- [x] Configurable solver parameters
- [x] Real-time result display

### 7. Browser Testing (Playwright)
- [x] Created `tests/bundle-adjustment.spec.ts` (11 tests)
- [x] Created `playwright.config.ts` with multi-browser support
- [x] Created `package.json` with test scripts
- [x] All tests passing in Chromium

### 8. CI/CD
- [x] Created `.github/workflows/test.yml`
- [x] Rust tests job
- [x] WASM build job with artifact upload
- [x] Playwright browser tests job

## File Reference

| File | Purpose | Status |
|------|---------|--------|
| `Cargo.toml` | Rust project config | DONE |
| `package.json` | Node.js/Playwright config | DONE |
| `playwright.config.ts` | Browser test config | DONE |
| `.cargo/config.toml` | WASM build flags | DONE |
| `.github/workflows/test.yml` | CI pipeline | DONE |
| `apex-solver-fork/Cargo.toml` | Fork with feature flags | DONE |
| `apex-solver-fork/src/lib.rs` | Feature-gated modules | DONE |
| `apex-solver-fork/src/core/problem.rs` | Conditional rayon | DONE |
| `apex-solver-fork/src/optimizer/*.rs` | web-time integration | DONE |
| `apex-solver-fork/src/error.rs` | Feature-gated IoError | DONE |
| `src/lib.rs` | WASM interface | DONE |
| `tests/bundle-adjustment.spec.ts` | Browser tests | DONE |
| `examples/index.html` | Browser demo | DONE |

## Technical Notes

### ReprojectionFactor Implementation

The custom `ReprojectionFactor` in `src/lib.rs`:
- Takes SE3 pose (7 params: tx, ty, tz, qw, qx, qy, qz) and 3D point (3 params)
- Transforms point from world to camera coordinates
- Projects using pinhole model with radial-tangential distortion
- Computes analytical Jacobians (2x9 matrix: 2x6 for pose, 2x3 for point)

### WASM Compatibility Fixes

1. **`std::time::Instant`** - Not available in WASM
   - Solution: Added `web-time` crate as dependency
   - Updated imports in `optimizer/mod.rs` and `optimizer/levenberg_marquardt.rs`

2. **`getrandom`** - Needs WASM backend configuration
   - Solution: Added `--cfg getrandom_backend="wasm_js"` to `.cargo/config.toml`
   - Also added `getrandom = { version = "0.3", features = ["wasm_js"] }` as direct dependency

3. **`rayon`** - No multi-threading in WASM
   - Solution: Feature-gated with `#[cfg(feature = "parallel")]`
   - Sequential fallback for WASM builds

4. **`memmap2`** - No file system in browser
   - Solution: Feature-gated with `#[cfg(feature = "io")]`

## Future Improvements

### Not Implemented (Out of Scope)
- [ ] Intrinsics optimization (focal length, principal point, distortion)
- [ ] Multi-threaded WASM (SharedArrayBuffer + Web Workers)
- [ ] Streaming/incremental optimization
- [ ] Covariance estimation output

### Potential Enhancements
- [ ] TypedArray input/output for better performance
- [ ] Progress callbacks during optimization
- [ ] Memory usage optimization
- [ ] Benchmark against other WASM BA implementations
