use lingua::LanguageDetectorBuilder;

pub use lingua::IsoCode639_1;
pub use lingua::Language;
pub struct LanguageDetector {
    detector: lingua::LanguageDetector,
}

impl LanguageDetector {
    pub fn new() -> Self {
        Self {
            detector: LanguageDetectorBuilder::from_all_languages()
                .with_preloaded_language_models()
                .build(),
        }
    }

    pub fn new_dev() -> Self {
        let langs = vec![Language::English, Language::Chinese, Language::Japanese];
        Self {
            detector: LanguageDetectorBuilder::from_languages(&langs)
                .with_preloaded_language_models()
                .build(),
        }
    }

    pub fn detect(&self, text: &str) -> Option<Language> {
        self.detector.detect_language_of(text)
    }

    pub fn detect_lang(&self, text: &str) -> String {
        match self.detect(text) {
            Some(lang) => lang.to_string(),
            None => "".to_string(),
        }
    }
}
