#![cfg(unix)] // Avoid running on Windows: the generated code will use `\r\n` instead of `\n`

use serde::Serialize;

static OUTPUT_PATH: &str = "tests/languages.json";

#[derive(Serialize)]
struct Lang(String, String, String, String);

#[test]
fn generated_language_list_if_outdated() {
    let languages = isolang::languages();
    let mut list: Vec<Lang> = Vec::new();
    for lg in languages {
        if lg.to_639_1().is_none() || lg.to_autonym().is_none() {
            continue;
        }
        let name = lg.to_name();
        if !name.is_ascii() {
            continue;
        }

        list.push(Lang(
            lg.to_639_1().unwrap().to_string(),
            lg.to_639_3().to_string(),
            lg.to_name().to_string(),
            lg.to_autonym().unwrap().to_string(),
        ));
    }

    list.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    let new_data = serde_json::to_string(&list).unwrap();

    let old_data = std::fs::read_to_string(OUTPUT_PATH).unwrap_or("".to_string());
    // write new output and fail test to draw attention
    if new_data != old_data {
        std::fs::write(OUTPUT_PATH, new_data).unwrap();

        panic!("generated data in the repository is outdated, updating...");
    }
}
