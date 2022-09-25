use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{anychar, satisfy},
    combinator::{recognize, map, value},
    multi::{many0, many1},
    sequence::{delimited, preceded},
    IResult,
};
use std::fs;
use xmlparser::{ElementEnd, Token, Tokenizer};

fn main() {
    println!("Reading enwik9");
    let text = fs::read_to_string("../data/enwik9").unwrap();
    println!("Parsing xml");
    let articles_w = parse_xml(&text);
    let mut articles = String::new();
    println!("Stripping wikitext");
    for (i,article_w) in articles_w.iter().enumerate() {
        if let Some(article) = strip_wikitext(article_w) {
            articles.push_str(&article);
            articles.push_str("\n########\n");
        }
        if i > 0 && i % 10000 == 0 {
            println!("{i}");
        }
    }
    fs::write("../data/stripped_enwik9.txt", articles).unwrap();
}

fn plain(i: &str) -> IResult<&str, &str> {
    recognize(many1(satisfy(
        |c| matches!(c, 'a'..='z' | 'A'..='Z' | '0'..='9' | ' ' | '\n' | ',' | '.' | '(' | ')' | ':' | '-' | '!' | '/'),
    )))(i)
}

fn link_contents(i: &str) -> IResult<&str, &str> {
    recognize(many1(satisfy(|c| !matches!(c, ']' | '|' | '['))))(i)
}

fn url_contents(i: &str) -> IResult<&str, &str> {
    recognize(many1(satisfy(|c| !matches!(c, ']' | ' ' | '['))))(i)
}

fn image_contents(i: &str) -> IResult<&str, &str> {
    value("", many1(satisfy(|c| !matches!(c, ']' | '['))))(i)
}

fn image_items(i: &str) -> IResult<&str, &str> {
    value("", many1(alt((image_contents, plain_link, named_link))))(i)
}

fn template_contents(i: &str) -> IResult<&str, &str> {
    value("", many1(satisfy(|c| !matches!(c, '}' | '{' | '&'))))(i)
}

fn template_items(i: &str) -> IResult<&str, &str> {
    value("", many1(alt((template_contents, template, table, nowiki, reference, entity))))(i)
}

fn bold_or_italic(i: &str) -> IResult<&str, &str> {
    value("", alt((tag("'''"), tag("''"))))(i)
}

fn heading(i: &str) -> IResult<&str, &str> {
    value("", preceded(tag("="), many1(tag("="))))(i)
}

fn entity(i: &str) -> IResult<&str, &str> {
    alt((
        value("&", tag("&amp;")),
        value("<", tag("&lt;")),
        value(">", tag("&gt;")),
        value("\"", tag("&quot;")),
    ))(i)
}

fn nowiki_contents(i: &str) -> IResult<&str, &str> {
    recognize(many1(satisfy(|c| !matches!(c, '&'))))(i)
}

fn nowiki(i: &str) -> IResult<&str, &str> {
    delimited(tag("&lt;nowiki&gt;"), nowiki_contents, tag("&lt;/nowiki&gt;"))(i)
}

fn reference(i: &str) -> IResult<&str, &str> {
    delimited(tag("&lt;ref&gt;"), nowiki_contents, tag("&lt;ref&gt;"))(i)
}

fn image(i: &str) -> IResult<&str, &str> {
    value("", delimited(alt((tag("[[image:"), tag("[[Image:"))), image_items, tag("]]")))(i)
}

fn plain_link(i: &str) -> IResult<&str, &str> {
    map(delimited(tag("[["), link_contents, tag("]]")), |text| if text.contains(":") {""} else {text})(i)
}

fn named_link(i: &str) -> IResult<&str, &str> {
    let (i, _) = tag("[[")(i)?;
    let (i, _) = link_contents(i)?;
    let (i, _) = tag("|")(i)?;
    let (i, result) = link_contents(i)?;
    let (i, _) = tag("]]")(i)?;
    Ok((i, result))
}

fn url(i: &str) -> IResult<&str, &str> {
    recognize(preceded(alt((tag("http://"), tag("https://"))), url_contents))(i)
}

fn url_link(i: &str) -> IResult<&str, &str> {
    let (i, _) = tag("[")(i)?;
    let (i, _) = url(i)?;
    let (i, _) = tag(" ")(i)?;
    let (i, result) = link_contents(i)?;
    let (i, _) = tag("]")(i)?;
    Ok((i, result))
}

fn template(i: &str) -> IResult<&str, &str> {
    value("", delimited(tag("{{"), template_items, tag("}}")))(i)
}

fn table(i: &str) -> IResult<&str, &str> {
    value("", delimited(tag("{|"), template_items, tag("}")))(i)
}

fn character(i: &str) -> IResult<&str, &str> {
    recognize(anychar)(i)
}

fn item(i: &str) -> IResult<&str, &str> {
    alt((plain, bold_or_italic, heading, nowiki, reference, entity, image, plain_link, named_link, url_link, template, table, character))(i)
}

fn strip_wikitext(input: &str) -> Option<String> {
    if input.starts_with("#REDIRECT ") {
        None
    } else {
        let mut result = String::new();
        if let Ok((_, items)) = many0(item)(input) {
            for item in &items {
                result.push_str(item);
            }
        }
        Some(result)
    }
}

fn parse_xml(text: &str) -> Vec<String> {
    let mut in_text = false;
    let mut articles = vec![];
    let mut article = String::new();
    for token in Tokenizer::from(&text as &str) {
        match token {
            Ok(Token::ElementStart { local, .. }) => {
                if local.as_str() == "text" {
                    in_text = true;
                }
            }
            Ok(Token::ElementEnd {
                end: ElementEnd::Close(.., local),
                ..
            }) => {
                if local.as_str() == "text" {
                    if in_text {
                        in_text = false;
                        articles.push(article.clone());
                        article = String::new();
                        if articles.len() % 10000 == 0 {
                            println!("{}", articles.len());
                        }
                    }
                }
            }
            Ok(Token::Text { text }) => {
                if in_text {
                    article.push_str(&text);
                }
            }
            _ => {}
        }
    }
    articles
}
