//! Structured outcome returned by each diagnostic case.

#[derive(Debug, Clone, PartialEq, Eq)]
/// Severity assigned to a diagnostic result.
pub enum DiagnosticStatus {
    Pass,
    Warning,
    Fail,
}

#[derive(Debug)]
/// Named status and human-readable measurements for one case.
pub struct DiagnosticResult {
    pub name: String,
    pub status: DiagnosticStatus,
    pub messages: Vec<String>,
}

impl DiagnosticResult {
    pub fn pass(name: &str) -> Self {
        Self {
            name: name.to_string(),
            status: DiagnosticStatus::Pass,
            messages: vec![],
        }
    }

    pub fn fail(name: &str, msg: &str) -> Self {
        Self {
            name: name.to_string(),
            status: DiagnosticStatus::Fail,
            messages: vec![msg.to_string()],
        }
    }

    pub fn warn(name: &str, msg: &str) -> Self {
        Self {
            name: name.to_string(),
            status: DiagnosticStatus::Warning,
            messages: vec![msg.to_string()],
        }
    }
}
