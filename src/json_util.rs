pub struct RawJSON {
    data: Vec<char>,
    offset: usize,
    result: String,
}

impl RawJSON {
    pub fn new(s: &str) -> Self {
        let s = s.trim();
        let cap = s.len();
        let data: Vec<char> = s.chars().collect();
        Self {
            data: data,
            offset: 0,
            result: String::with_capacity(cap),
        }
    }

    // 用于尝试修复 OpenAI translate 返回的 JSON String 无法解析 Vec<model::TEContent> 的问题
    // 仅需支持 array、object 和 string
    pub fn fix_me(mut self) -> Result<String, String> {
        self.skip_space();
        if self.offset == self.data.len() {
            return Err("no token to scan".to_string());
        }

        match self.data[self.offset] {
            '[' => {
                if let Some(s) = self.array() {
                    return Err(s);
                }
            }
            '{' => {
                if let Some(s) = self.object() {
                    return Err(s);
                }
            }
            '"' => {
                if let Some(s) = self.text() {
                    return Err(s);
                }
            }
            _ => {
                return Err(format!(
                    "unknown token `{}` to start fix_me",
                    self.data[self.offset]
                ));
            }
        }

        self.skip_space();
        if self.offset < self.data.len() {
            return Err(format!(
                "extraneous data exist: `{}`",
                self.data[self.offset]
            ));
        }

        return Ok(self.result);
    }

    fn skip_space(&mut self) {
        while self.offset < self.data.len() {
            if self.data[self.offset].is_whitespace() {
                self.offset += 1;
            } else {
                break;
            }
        }
    }

    fn array(&mut self) -> Option<String> {
        self.result.push('[');
        self.offset += 1;
        self.skip_space();

        if self.offset < self.data.len() && self.data[self.offset] == ']' {
            self.result.push(']');
            self.offset += 1;
            return None;
        }

        while self.offset < self.data.len() {
            match self.data[self.offset] {
                '{' => {
                    if let Some(s) = self.object() {
                        return Some(s);
                    }
                }
                '[' => {
                    if let Some(s) = self.array() {
                        return Some(s);
                    }
                }
                '"' => {
                    if let Some(s) = self.text() {
                        return Some(s);
                    }
                }
                _ => {
                    return Some(format!(
                        "unsupport token `{}{}` to start in array",
                        self.data[self.offset - 1],
                        self.data[self.offset]
                    ));
                }
            }

            self.skip_space();
            if self.offset >= self.data.len() {
                return Some("no token to scan in array".to_string());
            }

            match self.data[self.offset] {
                ',' => {
                    self.result.push(',');
                    self.offset += 1;
                    self.skip_space();
                }
                ']' => {
                    self.result.push(']');
                    self.offset += 1;
                    return None;
                }
                _ => {
                    return Some(format!(
                        "unsupport token `{}{}` to end in array",
                        self.data[self.offset - 1],
                        self.data[self.offset]
                    ));
                }
            }
        }

        return Some("no token to finish array".to_string());
    }

    fn object(&mut self) -> Option<String> {
        self.result.push('{');
        self.offset += 1;
        self.skip_space();

        if self.offset < self.data.len() && self.data[self.offset] == '}' {
            self.result.push('}');
            self.offset += 1;
            return None;
        }

        while self.offset < self.data.len() {
            // scan `key`
            match self.data[self.offset] {
                '"' => {
                    if let Some(s) = self.key() {
                        return Some(s);
                    }
                }
                _ => {
                    return Some(format!(
                        "unsupport token `{}{}` to start for object key",
                        self.data[self.offset - 1],
                        self.data[self.offset]
                    ));
                }
            }

            self.skip_space();
            if self.offset >= self.data.len() {
                return Some("no token to scan in object".to_string());
            }

            // scan `:`
            match self.data[self.offset] {
                ':' => {
                    self.result.push(':');
                    self.offset += 1;
                    self.skip_space();
                }
                _ => {
                    return Some(format!(
                        "unsupport token `{}{}` to start for object colon",
                        self.data[self.offset - 1],
                        self.data[self.offset]
                    ));
                }
            }

            self.skip_space();
            if self.offset >= self.data.len() {
                return Some("no token to scan in object".to_string());
            }

            // scan `value`
            match self.data[self.offset] {
                '{' => {
                    if let Some(s) = self.object() {
                        return Some(s);
                    }
                }
                '[' => {
                    if let Some(s) = self.array() {
                        return Some(s);
                    }
                }
                '"' => {
                    if let Some(s) = self.text() {
                        return Some(s);
                    }
                }
                _ => {
                    return Some(format!(
                        "unsupport token `{}{}` to start for object value",
                        self.data[self.offset - 1],
                        self.data[self.offset]
                    ));
                }
            }

            self.skip_space();
            if self.offset >= self.data.len() {
                return Some("no token to scan in object".to_string());
            }

            match self.data[self.offset] {
                ',' => {
                    self.result.push(',');
                    self.offset += 1;
                    self.skip_space();
                }
                '}' => {
                    self.result.push('}');
                    self.offset += 1;
                    return None;
                }
                _ => {
                    return Some(format!(
                        "unsupport token `{}{}` to end in object",
                        self.data[self.offset - 1],
                        self.data[self.offset]
                    ));
                }
            }
        }

        return Some("no char to scan in object".to_string());
    }

    fn can_not_end_text(&self) -> bool {
        let mut i = self.offset;
        let mut next_begin = false;
        while i < self.data.len() {
            if self.data[i].is_whitespace() {
                i += 1;
                continue;
            }

            if !next_begin {
                match self.data[i] {
                    '}' | ']' | ',' => {
                        next_begin = self.data[i] == ',';
                        i += 1;
                    }
                    _ => return true,
                }
            } else {
                match self.data[i] {
                    '"' | '{' | '[' => {
                        return false;
                    }
                    _ => return true,
                }
            }
        }
        return false;
    }

    fn key(&mut self) -> Option<String> {
        self.result.push('"');
        self.offset += 1;

        while self.offset < self.data.len() {
            match self.data[self.offset] {
                '"' => {
                    self.result.push('"');
                    self.offset += 1;
                    return None;
                }
                _ => {
                    self.result.push(self.data[self.offset]);
                    self.offset += 1;
                }
            }
        }

        return Some("no token to finish object key".to_string());
    }

    fn text(&mut self) -> Option<String> {
        self.result.push('"');
        self.offset += 1;

        while self.offset < self.data.len() {
            match self.data[self.offset] {
                '\\' => {
                    self.result.push('\\');
                    self.offset += 1;

                    if self.offset >= self.data.len() {
                        return Some("no token to scan for text".to_string());
                    }

                    match self.data[self.offset] {
                        '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u' => {
                            self.result.push(self.data[self.offset]);
                            self.offset += 1;
                        }
                        _ => {
                            // fix for a single `\`
                            self.result.push('\\');
                            self.result.push(self.data[self.offset]);
                            self.offset += 1;
                        }
                    }
                }
                '"' => {
                    self.offset += 1;
                    self.skip_space();
                    if self.offset >= self.data.len() {
                        self.result.push('"');
                        return None;
                    }

                    if self.can_not_end_text() {
                        // ignore an extra '"' and continue to scan
                        continue;
                    }

                    match self.data[self.offset] {
                        ',' | ':' | '}' | ']' => {
                            self.result.push('"');
                            return None;
                        }
                        _ => {
                            self.result.push(self.data[self.offset]);
                            self.offset += 1;
                        }
                    }
                }
                _ => {
                    self.result.push(self.data[self.offset]);
                    self.offset += 1;
                }
            }
        }

        return Some("no token to finish text".to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Case {
        input: String,
        output: String,
        err: Option<String>,
    }

    #[test]
    fn valid_json_works() {
        let test_cases: Vec<Case> = vec![
            Case {
                input: r#""""#.to_string(),
                output: r#""""#.to_string(),
                err: None,
            },
            Case {
                input: r#"" ""#.to_string(),
                output: r#"" ""#.to_string(),
                err: None,
            },
            Case {
                input: r#"" \"""#.to_string(),
                output: r#"" \"""#.to_string(),
                err: None,
            },
            Case {
                input: r#"{}"#.to_string(),
                output: r#"{}"#.to_string(),
                err: None,
            },
            Case {
                input: r#"{ } "#.to_string() + "\n",
                output: r#"{}"#.to_string(),
                err: None,
            },
            Case {
                input: r#"[]"#.to_string(),
                output: r#"[]"#.to_string(),
                err: None,
            },
            Case {
                input: r#"[ ] "#.to_string() + "\n",
                output: r#"[]"#.to_string(),
                err: None,
            },
            Case {
                input: r#"[
                    {
                      "id": "------",
                      "texts": []
                    },
                    {
                      "id": "Esp9G6",
                      "texts": [
                        "Stream:"
                      ]
                    },
                    {
                      "id": "------",
                      "texts": []
                    },
                    {
                      "id": "ykuRdu",
                      "texts": [
                        "Internet Engineering Task Force (IETF)"
                      ]
                    }
                ]"#
                .to_string(),
                output: r#"[{"id":"------","texts":[]},{"id":"Esp9G6","texts":["Stream:"]},{"id":"------","texts":[]},{"id":"ykuRdu","texts":["Internet Engineering Task Force (IETF)"]}]"#.to_string(),
                err: None,
            },
        ];

        for case in test_cases {
            match RawJSON::new(&case.input).fix_me() {
                Ok(val) => {
                    println!("FIX_OK: `{}` => `{}`, {}", case.input, val, val.len());
                    assert_eq!(val, case.output);
                }
                Err(err) => {
                    println!("FIX_ERR:  `{}` => `{}`", case.input, err);
                    assert!(case.err.is_some());
                    assert!(err.contains::<&str>(case.err.unwrap().as_ref()));
                }
            }
        }
    }

    #[test]
    fn fix_invalid_json_works() {
        let test_cases: Vec<Case> = vec![
            Case {
                input: r#"""""#.to_string(),
                output: r#""""#.to_string(),
                err: None,
            },
            Case {
                input: r#""" ""#.to_string(),
                output: r#""""#.to_string(),
                err: None,
            },
            Case {
                input: r#""\ ""#.to_string(),
                output: r#""\\ ""#.to_string(),
                err: None,
            },
            Case {
                input: r#"[
                    {
                      "id": "------",
                      "texts": []
                    },
                    {
                      "id": "Esp9G6",
                      "texts": [
                        ""] Stream: ["
                      ]
                    },
                    {
                      "id": "------",
                      "texts": []
                    },
                    {
                      "id": "ykuRdu",
                      "texts": [
                        "Internet Engineering Task Force (IETF)"
                      ]
                    }
                ]"#
                .to_string(),
                output: r#"[{"id":"------","texts":[]},{"id":"Esp9G6","texts":["] Stream: ["]},{"id":"------","texts":[]},{"id":"ykuRdu","texts":["Internet Engineering Task Force (IETF)"]}]"#.to_string(),
                err: None,
            },
            Case {
                input: r#"[{"id":"_Cu1P6","texts":["主类型2："]},{"id":"l0ZMCV","texts":["字节字符串。字符串中的字节数等于参数。例如，长度为5的字节字符串的初始字节为0b010_00101（主类型2，其他信息5表示长度），后跟5个二进制内容字节。长度为500的字节字符串将具有3个初始字节0b010_11001（主类型2，其他信息25表示两个字节的长度），后跟两个字节0x01f4，长度为500，后跟500个二进制内容字节。","¶"]},{"id":"A5CvEr","texts":["主类型3："]},{"id":"Jn1X6E","texts":["作为UTF-8 [","RFC3629",""]编码的文本字符串（","第2节", "）。字符串中的字节数等于参数。包含无效UTF-8序列的字符串是格式良好但无效的（","第1.2节", "）。此类型适用于需要解释或显示人类可读文本的系统，并允许区分结构化字节和具有指定曲目（Unicode）和编码（UTF-8）的文本。与JSON等格式不同，此类型中的Unicode字符永远不会被转义。因此，换行符（U+000A）始终表示为字符串中的字节0x0a，而不是字符0x5c6e（字符“\\”和“n”）或0x5c7530303061（字符“\\”，“u”，“0”，“0”，“0”和“a”）。","¶"]},{"id":"5NmwhW","texts":["主类型4："]},{"id":"a7YfwR","texts":["数据项数组。在其他格式中，数组也称为列表、序列或元组（“CBOR序列”是稍有不同的东西，但是[","RFC8742","]）。参数是数组中数据项的数量。数组中的项不需要全部是相同类型的。例如，包含10个任何类型的项的数组将具有初始字节0b100_01010（主类型4，其他信息10表示长度），后跟剩余的10个项。","¶"]},{"id":"4weExn","texts":["主类型5："]},{"id":"5Gt_2_","texts":["数据项对的映射。映射也称为表、字典、哈希或对象（在JSON中）。映射由数据项对组成，每个对由紧随其后的键和值组成。参数是映射中数据项对的数量。例如，包含9个对的映射将具有初始字节0b101_01001（主类型5，其他信息9表示对数），后跟剩余的18个项。第一项是第一个键，第二项是第一个值，第三项是第二个键，依此类推。由于映射中的项成对出现，它们的总数始终是偶数：包含奇数项的映射（在最后一个键数据项之后没有值数据项）不是格式良好的。具有重复键的映射可能是格式良好的，但它不是有效的，因此会导致不确定的解码；另请参见","第5.6节","。","¶"]},{"id":"sieJ4A","texts":["主类型6："]}]"#
                .to_string(),
                output: r#"[{"id":"_Cu1P6","texts":["主类型2："]},{"id":"l0ZMCV","texts":["字节字符串。字符串中的字节数等于参数。例如，长度为5的字节字符串的初始字节为0b010_00101（主类型2，其他信息5表示长度），后跟5个二进制内容字节。长度为500的字节字符串将具有3个初始字节0b010_11001（主类型2，其他信息25表示两个字节的长度），后跟两个字节0x01f4，长度为500，后跟500个二进制内容字节。","¶"]},{"id":"A5CvEr","texts":["主类型3："]},{"id":"Jn1X6E","texts":["作为UTF-8 [","RFC3629","]编码的文本字符串（","第2节","）。字符串中的字节数等于参数。包含无效UTF-8序列的字符串是格式良好但无效的（","第1.2节","）。此类型适用于需要解释或显示人类可读文本的系统，并允许区分结构化字节和具有指定曲目（Unicode）和编码（UTF-8）的文本。与JSON等格式不同，此类型中的Unicode字符永远不会被转义。因此，换行符（U+000A）始终表示为字符串中的字节0x0a，而不是字符0x5c6e（字符“\\”和“n”）或0x5c7530303061（字符“\\”，“u”，“0”，“0”，“0”和“a”）。","¶"]},{"id":"5NmwhW","texts":["主类型4："]},{"id":"a7YfwR","texts":["数据项数组。在其他格式中，数组也称为列表、序列或元组（“CBOR序列”是稍有不同的东西，但是[","RFC8742","]）。参数是数组中数据项的数量。数组中的项不需要全部是相同类型的。例如，包含10个任何类型的项的数组将具有初始字节0b100_01010（主类型4，其他信息10表示长度），后跟剩余的10个项。","¶"]},{"id":"4weExn","texts":["主类型5："]},{"id":"5Gt_2_","texts":["数据项对的映射。映射也称为表、字典、哈希或对象（在JSON中）。映射由数据项对组成，每个对由紧随其后的键和值组成。参数是映射中数据项对的数量。例如，包含9个对的映射将具有初始字节0b101_01001（主类型5，其他信息9表示对数），后跟剩余的18个项。第一项是第一个键，第二项是第一个值，第三项是第二个键，依此类推。由于映射中的项成对出现，它们的总数始终是偶数：包含奇数项的映射（在最后一个键数据项之后没有值数据项）不是格式良好的。具有重复键的映射可能是格式良好的，但它不是有效的，因此会导致不确定的解码；另请参见","第5.6节","。","¶"]},{"id":"sieJ4A","texts":["主类型6："]}]"#.to_string(),
                err: None,
            },
        ];

        for case in test_cases {
            match RawJSON::new(&case.input).fix_me() {
                Ok(val) => {
                    println!("FIX_OK: `{}` => `{}`, {}", case.input, val, val.len());
                    assert_eq!(val, case.output);
                }
                Err(err) => {
                    println!("FIX_ERR:  `{}` => `{}`", case.input, err);
                    assert!(case.err.is_some());
                    assert!(err.contains::<&str>(case.err.unwrap().as_ref()));
                }
            }
        }
    }
}
