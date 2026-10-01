#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagnosticStatus {
    Pass,
    Warning,
    Fail,
}

#[derive(Debug)]
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