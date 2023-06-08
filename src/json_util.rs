pub struct RawJSONArray {
    chars: Vec<char>,
    offset: usize,
    result: Vec<char>,
}

impl RawJSONArray {
    pub fn new(s: &str) -> Self {
        let s = s.trim();
        let cap = s.len();
        let chars: Vec<char> = s.chars().collect();
        Self {
            chars,
            offset: 0,
            result: Vec::with_capacity(cap),
        }
    }

    // 用于尝试修复 OpenAI translate 返回的 JSON String 无法解析 Vec<Vec<String>> 的问题
    pub fn fix_me(mut self) -> Result<String, String> {
        self.skip_space();
        if self.offset >= self.chars.len() {
            return Err("no token to scan".to_string());
        }

        match self.chars[self.offset] {
            '[' => {
                if let Some(s) = self.array() {
                    return Err(s);
                }
            }
            _ => {
                return Err(format!(
                    "unknown token `{}` to start fix_me",
                    self.chars[self.offset]
                ));
            }
        }

        self.skip_space();
        if self.offset < self.chars.len() {
            return Err(format!(
                "extraneous data exist: `{}`",
                self.chars[self.offset]
            ));
        }

        Ok(String::from_iter(&self.result))
    }

    fn skip_space(&mut self) {
        while self.offset < self.chars.len() {
            if self.chars[self.offset].is_whitespace() {
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

        if self.offset < self.chars.len() && self.chars[self.offset] == ']' {
            self.result.push(']');
            self.offset += 1;
            return None;
        }

        while self.offset < self.chars.len() {
            match self.chars[self.offset] {
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
                    // case: miss a '"'
                    if self.result.last() == Some(&',') && self.result[self.result.len() - 2] == '"'
                    {
                        self.offset -= 1;
                        if let Some(s) = self.text() {
                            return Some(s);
                        }
                    } else {
                        return Some(format!(
                            "unsupport token `{}{}` at {} to start in array",
                            self.chars[self.offset - 1],
                            self.chars[self.offset],
                            self.offset
                        ));
                    }
                }
            }

            self.skip_space();
            if self.offset >= self.chars.len() {
                return Some("no token to scan in array".to_string());
            }

            match self.chars[self.offset] {
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
                        self.chars[self.offset - 1],
                        self.chars[self.offset]
                    ));
                }
            }
        }

        Some("no token to finish array".to_string())
    }

    fn can_not_end_text(&self) -> bool {
        let mut i = self.offset;
        while i < self.chars.len() {
            if self.chars[i].is_whitespace() {
                i += 1;
                continue;
            }
            match self.chars[i] {
                ',' => return false,
                ']' => {
                    i += 1;
                }
                _ => return true,
            }
        }
        false
    }

    fn text(&mut self) -> Option<String> {
        self.result.push('"');
        self.offset += 1;

        while self.offset < self.chars.len() {
            match self.chars[self.offset] {
                '\\' => {
                    self.result.push('\\');
                    self.offset += 1;

                    if self.offset >= self.chars.len() {
                        return Some("no token to scan for text".to_string());
                    }

                    match self.chars[self.offset] {
                        '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u' => {
                            self.result.push(self.chars[self.offset]);
                            self.offset += 1;
                        }
                        _ => {
                            // case: miss a '\'
                            self.result.push('\\');
                            self.result.push(self.chars[self.offset]);
                            self.offset += 1;
                        }
                    }
                }
                '"' => {
                    self.offset += 1;
                    self.skip_space();
                    if self.offset >= self.chars.len() {
                        self.result.push('"');
                        return None;
                    }

                    if self.can_not_end_text() {
                        // case: ignore an extra '"' and continue to scan
                        continue;
                    }

                    match self.chars[self.offset] {
                        ',' | ']' => {
                            self.result.push('"');
                            return None;
                        }
                        _ => {
                            self.result.push(self.chars[self.offset]);
                            self.offset += 1;
                        }
                    }
                }
                _ => {
                    self.result.push(self.chars[self.offset]);
                    self.offset += 1;
                }
            }
        }

        Some("no token to finish text".to_string())
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
                input: r#"[""]"#.to_string(),
                output: r#"[""]"#.to_string(),
                err: None,
            },
            Case {
                input: r#"[" "]"#.to_string(),
                output: r#"[" "]"#.to_string(),
                err: None,
            },
            Case {
                input: r#"[" \""]"#.to_string(),
                output: r#"[" \""]"#.to_string(),
                err: None,
            },
            Case {
                input: r#"[
                    [],
                    [
                        "Stream:"
                    ],
                    [],
                    [
                        "texts",
                        "Internet Engineering Task Force (IETF)"
                    ]
                ]"#
                .to_string(),
                output: r#"[[],["Stream:"],[],["texts","Internet Engineering Task Force (IETF)"]]"#
                    .to_string(),
                err: None,
            },
        ];

        for case in test_cases {
            match RawJSONArray::new(&case.input).fix_me() {
                Ok(val) => {
                    // println!("FIX_OK: `{}` => `{}`, {}", case.input, val, val.len());
                    assert_eq!(val, case.output);
                }
                Err(err) => {
                    // println!("FIX_ERR:  `{}` => `{}`", case.input, err);
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
                input: r#"["""]"#.to_string(),
                output: r#"[""]"#.to_string(),
                err: None,
            },
            Case {
                input: r#"["" "]"#.to_string(),
                output: r#"[""]"#.to_string(),
                err: None,
            },
            Case {
                input: r#"["\ "]"#.to_string(),
                output: r#"["\\ "]"#.to_string(),
                err: None,
            },
            Case {
                input: r#"[
                    [],
                    [
                        ""] Stream: ["
                    ],
                    [
                        "Internet Engineering Task Force \(IETF)"
                    ]
                ]"#
                .to_string(),
                output: r#"[[],["] Stream: ["],["Internet Engineering Task Force \\(IETF)"]]"#.to_string(),
                err: None,
            },
            Case {
                input: r#"[["作为UTF-8 [","RFC3629",""]编码的文本字符串（","第2节", "）。字符串中的字节数等于参数。包含无效UTF-8序列的字符串是格式良好但无效的（","第1.2节", "）。此类型适用于需要解释或显示人类可读文本的系统，并允许区分结构化字节和具有指定曲目（Unicode）和编码（UTF-8）的文本。与JSON等格式不同，此类型中的Unicode字符永远不会被转义。因此，换行符（U+000A）始终表示为字符串中的字节0x0a，而不是字符0x5c6e（字符“\”和“n”）或0x5c7530303061（字符“\”，“u”，“0”，“0”，“0”和“a”）。","¶"]]"#
                .to_string(),
                output: r#"[["作为UTF-8 [","RFC3629","]编码的文本字符串（","第2节","）。字符串中的字节数等于参数。包含无效UTF-8序列的字符串是格式良好但无效的（","第1.2节","）。此类型适用于需要解释或显示人类可读文本的系统，并允许区分结构化字节和具有指定曲目（Unicode）和编码（UTF-8）的文本。与JSON等格式不同，此类型中的Unicode字符永远不会被转义。因此，换行符（U+000A）始终表示为字符串中的字节0x0a，而不是字符0x5c6e（字符“\\”和“n”）或0x5c7530303061（字符“\\”，“u”，“0”，“0”，“0”和“a”）。","¶"]]"#.to_string(),
                err: None,
            },
            Case {
                input: r#"[["特定数据模型还可以为映射键和编码器自由度指定值等效性（包括不同类型的值）。例如，在通用数据模型中，有效的映射可以同时具有 ","0",", ","0.0",", 作为键，并且编码器不得将 ","0.0", 编码为整数（主类型 0， ","第 3.1 节",）。但是，如果特定数据模型声明整数值和浮点表示的整数值等效，则在单个映射中使用两个映射键 ","0",", ","0.0",", 将被视为重复，即使它们被编码为不同的主类型，因此无效；编码器可以将整数值的浮点数编码为整数或反之亦然，可能是为了节省编码字节。","¶"],["3. ","CBOR 编码的规范"],["CBOR 数据项（","第 2 节",") 被编码为或从携带有形式良好的编码数据项的字节字符串中解码，如本节所述。编码总结在 ","附录 B"," 中的 ","表 7"," 中，由初始字节索引。编码器必须仅生成形式良好的编码数据项。当解码器遇到不是形式良好的编码 CBOR 数据项的输入时，解码器不得返回已解码的数据项（这并不影响可能提供一些来自损坏的编码 CBOR 数据项的信息的诊断和恢复工具的有用性）。","¶"],["每个编码数据项的初始字节都包含有关主类型（高 3 位，如 ","第 3.1 节"," 中所述）和其他信息（低 5 位）的信息。除了少数例外，附加信息的值描述如何加载无符号整数“参数”：","¶"],["小于 24："],["参数的值是附加信息的值。","¶"],["24、25、26 或 27："],["参数的值分别保存在以下 1、2、4 或 8 个字节中，以网络字节顺序排列。对于主类型 7 和附加信息值 25、26、27，这些字节不用作整数参数，而用作浮点值（请参见 ","第 3.3 节",）。","¶"],["28、29、30："],["这些值保留用于将来添加到 CBOR 格式中。在 CBOR 的当前版本中，编码项不是形式良好的。","¶"],["31："],["不派生参数值。如果主类型为 0、1 或 6，则编码项不是形式良好的。对于主类型 2 到 5，项目的长度是不确定的，对于主类型 7，字节根本不构成数据项，而是终止无限长度项；所有这些都在 ","第 3.2 节"," 中描述。","¶"],["编码数据项的初始字节和任何其他字节用于构造参数的集合称为数据项的头部。","¶"],["此参数的含义取决于主类型。例如，在主类型 0 中，参数是数据项本身的值（在主类型 1 中，数据项的值是从参数计算出的）；在主类型 2 和 3 中，它给出了随后的字符串数据的字节长度；在主类型 4 和 5 中，它用于确定所包含的数据项的数量。","¶"],["如果编码的字节序列在数据项结束之前结束，则该项不是形式良好的。如果编码的字节序列在最外层编码项解码后仍有剩余字节，则该编码不是单个形式良好的 CBOR 项。根据应用程序，解码器可以将编码视为不是形式良好的，或者仅将剩余字节的开始标识给应用程序。","¶"],["CBOR 解码器实现可以基于具有初始字节的所有 256 个定义值的跳转表（","表 7",）。约束实现中的解码器可以使用初始字节和后续字节的结构进行更紧凑的代码（有关此代码的大致印象，请参见 ","附录 C","）。","¶"],["3.1. ","主类型"],["以下列出了主类型及其关联的附加信息和其他字节。","¶"],["主类型 0："],["范围在 0..2","64","-1 内的无符号整数。编码项的值是参数本身。例如，整数 10 表示为一个字节 0b000_01010（主类型 0，附加信息 10）。整数 500 将是 0b000_11001（主类型 0，附加信息 25）后跟两个字节 0x01f4，即十进制中的 500。","¶"],["主类型 1："],["范围在 -2","64","..-1 内的负整数。项目的值为 -1 减去参数。例如，整数 -500 将是 0b001_11001（主类型 1，附加信息 25）后跟两个字节 0x01f3，即十进制中的 499。","¶"]]"#.to_string(),
                output: r#"[["特定数据模型还可以为映射键和编码器自由度指定值等效性（包括不同类型的值）。例如，在通用数据模型中，有效的映射可以同时具有 ","0",", ","0.0",", 作为键，并且编码器不得将 ","0.0","编码为整数（主类型 0， ","第 3.1 节","）。但是，如果特定数据模型声明整数值和浮点表示的整数值等效，则在单个映射中使用两个映射键 ","0",", ","0.0",", 将被视为重复，即使它们被编码为不同的主类型，因此无效；编码器可以将整数值的浮点数编码为整数或反之亦然，可能是为了节省编码字节。","¶"],["3. ","CBOR 编码的规范"],["CBOR 数据项（","第 2 节",") 被编码为或从携带有形式良好的编码数据项的字节字符串中解码，如本节所述。编码总结在 ","附录 B"," 中的 ","表 7"," 中，由初始字节索引。编码器必须仅生成形式良好的编码数据项。当解码器遇到不是形式良好的编码 CBOR 数据项的输入时，解码器不得返回已解码的数据项（这并不影响可能提供一些来自损坏的编码 CBOR 数据项的信息的诊断和恢复工具的有用性）。","¶"],["每个编码数据项的初始字节都包含有关主类型（高 3 位，如 ","第 3.1 节"," 中所述）和其他信息（低 5 位）的信息。除了少数例外，附加信息的值描述如何加载无符号整数“参数”：","¶"],["小于 24："],["参数的值是附加信息的值。","¶"],["24、25、26 或 27："],["参数的值分别保存在以下 1、2、4 或 8 个字节中，以网络字节顺序排列。对于主类型 7 和附加信息值 25、26、27，这些字节不用作整数参数，而用作浮点值（请参见 ","第 3.3 节","）。","¶"],["28、29、30："],["这些值保留用于将来添加到 CBOR 格式中。在 CBOR 的当前版本中，编码项不是形式良好的。","¶"],["31："],["不派生参数值。如果主类型为 0、1 或 6，则编码项不是形式良好的。对于主类型 2 到 5，项目的长度是不确定的，对于主类型 7，字节根本不构成数据项，而是终止无限长度项；所有这些都在 ","第 3.2 节"," 中描述。","¶"],["编码数据项的初始字节和任何其他字节用于构造参数的集合称为数据项的头部。","¶"],["此参数的含义取决于主类型。例如，在主类型 0 中，参数是数据项本身的值（在主类型 1 中，数据项的值是从参数计算出的）；在主类型 2 和 3 中，它给出了随后的字符串数据的字节长度；在主类型 4 和 5 中，它用于确定所包含的数据项的数量。","¶"],["如果编码的字节序列在数据项结束之前结束，则该项不是形式良好的。如果编码的字节序列在最外层编码项解码后仍有剩余字节，则该编码不是单个形式良好的 CBOR 项。根据应用程序，解码器可以将编码视为不是形式良好的，或者仅将剩余字节的开始标识给应用程序。","¶"],["CBOR 解码器实现可以基于具有初始字节的所有 256 个定义值的跳转表（","表 7","）。约束实现中的解码器可以使用初始字节和后续字节的结构进行更紧凑的代码（有关此代码的大致印象，请参见 ","附录 C","）。","¶"],["3.1. ","主类型"],["以下列出了主类型及其关联的附加信息和其他字节。","¶"],["主类型 0："],["范围在 0..2","64","-1 内的无符号整数。编码项的值是参数本身。例如，整数 10 表示为一个字节 0b000_01010（主类型 0，附加信息 10）。整数 500 将是 0b000_11001（主类型 0，附加信息 25）后跟两个字节 0x01f4，即十进制中的 500。","¶"],["主类型 1："],["范围在 -2","64","..-1 内的负整数。项目的值为 -1 减去参数。例如，整数 -500 将是 0b001_11001（主类型 1，附加信息 25）后跟两个字节 0x01f3，即十进制中的 499。","¶"]]"#.to_string(),
                err: None,
            },
        ];

        for case in test_cases {
            match RawJSONArray::new(&case.input).fix_me() {
                Ok(val) => {
                    // println!("FIX_OK: `{}` => `{}`, {}", case.input, val, val.len());
                    assert_eq!(val, case.output);
                }
                Err(err) => {
                    // println!("FIX_ERR:  `{}` => `{}`", case.input, err);
                    assert!(case.err.is_some());
                    assert!(err.contains::<&str>(case.err.unwrap().as_ref()));
                }
            }
        }
    }
}
