//! Measurement-only noise probe (`CBS_NOISE_PROBE=1`, cargo feature
//! `noise-audit`).
//!
//! The probe decrypts intermediate ciphertexts with the harness secret key and
//! records the signed phase residual of every product digit and every
//! normalizer PBS input, so the additive-variance model of the failure
//! analysis can be compared against measured noise.  When the harness supplies
//! expected plaintext values (propagated through the public schedule from the
//! clear operands), residuals are measured against the true expected slot and
//! a `slot_mismatch` flag records any value that decoded to a different slot;
//! without expected values the nearest encoding multiple is used, which is
//! adequate for variance estimation but cannot witness slot-crossing errors.
//!
//! This is an audit tool: the release build (without the `noise-audit`
//! feature) contains no decryption path — `from_env` refuses `CBS_NOISE_PROBE`
//! and the record path is not compiled.  The secret key stays inside the
//! harness closure that the evaluator invokes blindly.

use tfhe::core_crypto::prelude::*;

/// (stage, wave, column, event, bound, terms, unrefreshed, expected, ciphertext).
pub(crate) type NoiseProbeFn<'a> = &'a (dyn Fn(&str, usize, usize, usize, u64, usize, usize, Option<u64>, &LweCiphertextOwned<u64>)
         + Sync);

#[cfg(feature = "noise-audit")]
mod enabled {
    use crate::config::{encoding_delta, env_flag};
    use crate::keys::{raw_phase, HarnessSecretKeys};
    use anyhow::{Context, Result};
    use std::env;
    use std::fs::{create_dir_all, OpenOptions};
    use std::io::Write;
    use std::path::Path;
    use std::sync::Mutex;
    use tfhe::core_crypto::prelude::*;

    pub(crate) struct NoiseProbe {
        label: String,
        width_bits: usize,
        run_id: String,
        writer: Mutex<std::fs::File>,
    }

    const PROBE_BITS: usize = 4;

    impl NoiseProbe {
        pub(crate) fn from_env(label: &str, width_bits: usize) -> Result<Option<Self>> {
            if !env_flag("CBS_NOISE_PROBE") {
                return Ok(None);
            }
            let path = env::var("CBS_NOISE_PROBE_CSV")
                .unwrap_or_else(|_| "results/noise-probe.csv".to_string());
            if let Some(parent) = Path::new(&path).parent() {
                if !parent.as_os_str().is_empty() {
                    create_dir_all(parent)
                        .with_context(|| format!("failed to create probe directory for {path}"))?;
                }
            }
            let write_header = std::fs::metadata(&path)
                .map(|metadata| metadata.len() == 0)
                .unwrap_or(true);
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .with_context(|| format!("failed to open CBS_NOISE_PROBE_CSV={path}"))?;
            if write_header {
                writeln!(
                    file,
                    "run_id,label,width_bits,stage,wave,column,event,bound,terms,unrefreshed,expected,decoded,residual,slot_mismatch"
                )?;
            }
            Ok(Some(Self {
                label: label.to_string(),
                width_bits,
                run_id: env::var("CBS_NOISE_PROBE_RUN_ID")
                    .unwrap_or_else(|_| std::process::id().to_string()),
                writer: Mutex::new(file),
            }))
        }

        /// Record one clean term or chunk sum.  With an expected value the
        /// residual is measured against the true slot and slot crossings are
        /// flagged; otherwise the nearest encoding multiple is used.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn record(
            &self,
            secrets: &HarnessSecretKeys,
            stage: &str,
            wave: usize,
            column: usize,
            event: usize,
            bound: u64,
            terms: usize,
            unrefreshed: usize,
            expected: Option<u64>,
            ciphertext: &LweCiphertextOwned<u64>,
        ) {
            let raw = raw_phase(ciphertext, secrets);
            let delta = encoding_delta(PROBE_BITS);
            let nearest = raw.wrapping_add(delta / 2) / delta;
            let decoded = nearest % (1u64 << (PROBE_BITS + 1));
            let (reference_slot, mismatch) = match expected {
                Some(value) => (value, Some(decoded != value)),
                None => (decoded, None),
            };
            let residual = raw.wrapping_sub(reference_slot.wrapping_mul(delta)) as i64;
            let mut writer = self.writer.lock().expect("noise-probe writer poisoned");
            writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                self.run_id,
                self.label,
                self.width_bits,
                stage,
                wave,
                column,
                event,
                bound,
                terms,
                unrefreshed,
                expected.map_or_else(String::new, |value| value.to_string()),
                decoded,
                residual,
                mismatch.map_or_else(String::new, |flag| u8::from(flag).to_string()),
            )
            .expect("noise-probe row write failed");
        }
    }
}

#[cfg(feature = "noise-audit")]
pub(crate) use enabled::NoiseProbe;

#[cfg(not(feature = "noise-audit"))]
mod disabled {
    use crate::config::env_flag;
    use crate::keys::HarnessSecretKeys;
    use anyhow::{bail, Result};
    use tfhe::core_crypto::prelude::*;

    /// Release-build stub: the probe cannot be constructed, so no decryption
    /// callback exists in normal benchmark binaries.
    pub(crate) struct NoiseProbe;

    impl NoiseProbe {
        pub(crate) fn from_env(_label: &str, _width_bits: usize) -> Result<Option<Self>> {
            if env_flag("CBS_NOISE_PROBE") {
                bail!(
                    "CBS_NOISE_PROBE=1 requires a build with --features noise-audit; \
                     the release binary contains no decryption path"
                );
            }
            Ok(None)
        }

        #[allow(clippy::too_many_arguments)]
        pub(crate) fn record(
            &self,
            _secrets: &HarnessSecretKeys,
            _stage: &str,
            _wave: usize,
            _column: usize,
            _event: usize,
            _bound: u64,
            _terms: usize,
            _unrefreshed: usize,
            _expected: Option<u64>,
            _ciphertext: &LweCiphertextOwned<u64>,
        ) {
            unreachable!("noise probe is not constructible without the noise-audit feature");
        }
    }
}

#[cfg(not(feature = "noise-audit"))]
pub(crate) use disabled::NoiseProbe;
