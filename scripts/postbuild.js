#!/usr/bin/env node
/**
 * Post-build script for sba-solver-wasm
 *
 * This script runs after wasm-pack build to:
 * 1. Copy wrapper.js to pkg/
 * 2. Merge pkg-extras.json into pkg/package.json
 * 3. Add wrapper.js to the files array
 */

import { readFileSync, writeFileSync, copyFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..');

const pkgDir = join(rootDir, 'pkg');
const jsDir = join(rootDir, 'js');

// Check that pkg/ exists (wasm-pack should have created it)
if (!existsSync(pkgDir)) {
    console.error('Error: pkg/ directory does not exist. Run wasm-pack build first.');
    process.exit(1);
}

// 1. Copy wrapper.js to pkg/
const wrapperSrc = join(jsDir, 'wrapper.js');
const wrapperDst = join(pkgDir, 'wrapper.js');

if (!existsSync(wrapperSrc)) {
    console.error('Error: js/wrapper.js does not exist.');
    process.exit(1);
}

copyFileSync(wrapperSrc, wrapperDst);
console.log('Copied js/wrapper.js -> pkg/wrapper.js');

// 2. Read and merge package.json with extras
const pkgJsonPath = join(pkgDir, 'package.json');
const extrasPath = join(jsDir, 'pkg-extras.json');

if (!existsSync(pkgJsonPath)) {
    console.error('Error: pkg/package.json does not exist.');
    process.exit(1);
}

const pkg = JSON.parse(readFileSync(pkgJsonPath, 'utf-8'));
const extras = JSON.parse(readFileSync(extrasPath, 'utf-8'));

// Merge extras into pkg
Object.assign(pkg, extras);

// 3. Add wrapper.js to files array if not already present
if (!pkg.files) {
    pkg.files = [];
}
if (!pkg.files.includes('wrapper.js')) {
    pkg.files.push('wrapper.js');
}

// Write updated package.json
writeFileSync(pkgJsonPath, JSON.stringify(pkg, null, 2) + '\n');
console.log('Updated pkg/package.json with exports and wrapper.js');

console.log('Post-build complete!');
