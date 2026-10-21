"""Text rules, on synthetic strings shaped like recognizer output."""

from indic_ocr_harness.text import (
    LATEX_RE,
    agreed_edits,
    ban_chars,
    canon_date,
    contained_in,
    count_numbers,
    cut_macro_repeat,
    dates_in,
    degenerate,
    foreign_letters,
    harvest_filter,
    latex_spans,
    latex_to_plain,
    loop_tail,
    looped,
    minority_digits,
    splice,
    strip_degenerate_lines,
    suspect_latex,
    to_script,
    trim_loop,
    word_spans,
)
from indic_ocr_harness.text.scripts import DEVA_DIGIT


def test_latex_to_plain():
    assert latex_to_plain(r"खेत $\frac{12}{7}$ और $\frac{12}{3}$") == "खेत 12/7 और 12/3"
    assert latex_to_plain(r"$ \text{कुल} = 18 \quad 4200 $") == "कुल = 18 4200"
    assert latex_to_plain(r"$$254810$$") == "254810"
    assert latex_to_plain("कोई गणित नहीं") == "कोई गणित नहीं"
    assert latex_to_plain(r"$14x=3x-6$") == "14x=3x-6"  # not recoverable by rewriting: the reason for the latex ban


def test_suspect_latex():
    assert suspect_latex("$14x=3x-6$") and suspect_latex("$BD$")
    assert not suspect_latex(r"$\frac{12}{5}$") and not suspect_latex(r"$ \text{कुल} = 18 \quad 4200 $")
    assert not suspect_latex("$$254810$$")


def test_gates():
    assert foreign_letters("राम શ્યામ", "deva") == (5, 8)
    assert foreign_letters("राम। श्याम॥", "deva") == (0, 8)
    assert minority_digits("640 720 640 720 ६४०") == "deva"
    assert minority_digits("१२/४ ३.५०२ ४.११७ 0.215 2.07 ९५/२ ०.८४१ २१/३ ४४/१ २.३०६ १.५२८ ३.०९४ १७/११८/२") == "ascii"
    assert minority_digits("640 720 1") is None  # a single stray digit is not evidence
    assert minority_digits("12 १२") is None  # evenly mixed: not a small minority
    assert minority_digits("मु० 45/1 मु० 45/3 मु० 46/2 न० 210") is None  # ० after a letter is an abbreviation mark
    assert LATEX_RE.search(r"दिनांक $14x=3x-6$") and not LATEX_RE.search("दिनांक 14-3-26")


def test_to_script():
    assert to_script("राम શ્યામ") == "राम श्याम"
    assert to_script("বাজার", "deva") == "बाजार"
    assert to_script("बाजार", "beng") == "বাজার"
    assert len(to_script("ગામ ਪਿੰਡ ଗାଁ")) == len("ગામ ਪਿੰਡ ଗାଁ")


def test_ban_chars():
    assert ban_chars("latex") == {"$", "\\"}
    s = ban_chars("script:deva")
    assert "ક" in s and "ক" in s and "क" not in s and "।" not in s
    b = ban_chars("script:beng")
    assert "क" in b and "ক" not in b and "।" not in b
    assert ban_chars("digits:deva") == set("०१२३४५६७८९") and ban_chars("digits:ascii") == set("0123456789")


def test_splice_and_vote():
    a = "दिनांक $14x=3x-6$ को बैठक हुई"
    b = "दिनांक 14-3-26 को बेठक हुई"  # the re-read also flips a letter elsewhere
    assert splice(a, b, latex_spans(a)) == "दिनांक 14-3-26 को बैठक हुई"
    a2 = "640 15 720 ६४० 15 720"
    b2 = "640 15 720 640 15 720"
    assert splice(a2, b2, word_spans(a2, lambda w: bool(DEVA_DIGIT.search(w)))) == b2
    assert splice("abc def", "xyz uvw", []) == "abc def"
    assert (
        splice("राम શ્યામ आया", "राम श्याम आया", word_spans("राम શ્યામ आया", lambda w: foreign_letters(w)[0] > 0))
        == "राम श्याम आया"
    )
    # two-of-three vote: only the edit both views make is applied
    new, n = agreed_edits("बश्ती बाई 4510", "बस्ती बाई 4510", "बस्ती बाइ 4610")
    assert new == "बस्ती बाई 4510" and n == 1
    assert agreed_edits("abc", "abd", "abe") == ("abc", 0)


def test_spans():
    assert latex_spans(r"a $x$ b \\frac{1}{2} c") == [(2, 5), (8, 20)]


def test_degenerate():
    junk = "प्र-प्र-प्र-प्र-प्र-प्र-प्रप्र-प्रप्रप्रप्रप्रप्रप्रप्र-प्रप्रप्र-प्र-" * 30
    assert degenerate(junk) and degenerate("प्रमममम-मममम-प्रमममम-" * 40)
    prose = (
        "आज दिनांक 15/9/2026 को ग्राम संगठन की बैठक हुई जिसमें सभी सदस्यों ने भाग लिया और बचत की राशि जमा की गई। "
        "बैठक में यह निर्णय लिया गया कि अगली बैठक में ऋण वापसी पर चर्चा होगी तथा कोषाध्यक्ष हिसाब प्रस्तुत करेंगी।"
    )
    assert not degenerate(prose)
    register = (
        "<table>"
        + "".join(f"<tr><td>{i}</td><td>नमूना सदस्य</td><td>640</td><td>15</td><td>720</td></tr>" for i in range(1, 15))
        + "</table>"
    )
    assert not degenerate(register, "table")  # identical rows with incrementing serials
    assert degenerate("<table>" + "<tr><td>7</td><td>640</td><td>15</td></tr>" * 6 + "</table>", "table")
    assert strip_degenerate_lines("पहली पंक्ति ठीक है\n" + junk) == "पहली पंक्ति ठीक है"


def test_loops():
    assert looped("बैठक " + "abcd" * 4) and not looped("बैठक abcdabcdabcd")
    assert trim_loop("नाम: " + "राम " * 10) == "नाम: राम "
    assert trim_loop("no loop here") == "no loop here"
    item = "प्रस्ताव संख्या एक: समूह की बचत राशि बैंक में जमा की जाएगी और हिसाब रखा जाएगा"
    assert cut_macro_repeat(f"1. {item}\n2. दूसरा\n1. {item}\n2. दूसरा") == f"1. {item}\n2. दूसरा"


def test_loop_tail():
    """The online loop stop fires only on long loops of real content, not on fill lines or repeated table markup."""
    assert loop_tail("बैठक की कार्यवाही " + "अनाज / दाल / " * 20)
    assert not loop_tail("अनाज / " * 5)  # a loop, but the repeated tail is under 200 characters
    assert not loop_tail("नाम " + "<td></td>" * 60)  # empty cells are legitimate
    assert not loop_tail("दिनांक " + "." * 300)  # a fill line of 300 dots is kept
    assert loop_tail("दिनांक " + "." * 600)  # ... 600 is a loop
    assert not loop_tail("आज दिनांक 15/9/2026 को ग्राम सभा की बैठक हुई, जिसमें सभी सदस्य उपस्थित थे।")


def test_contained_in():
    """The akshara/number duplicate test on outside-grid regions."""
    ak, nm = contained_in(
        "01. उत्तर खंड नहर-2210 418.25",
        "<table><tr><td>01.</td><td>उत्तर खंड</td><td>नहर-2210</td><td>418.25</td></tr></table>",
    )
    assert nm == 1.0 and ak >= 0.5
    ak, nm = contained_in("दिनांक 11-6-2025 को ग्राम सभा की बैठक हुई", "<table><tr><td>नाम</td><td>राशि 500</td></tr></table>")
    assert ak < 0.5 and nm == 0.0  # one number only: not judged on numbers
    assert contained_in("", "<table></table>") == (0.0, 0.0)


def test_dates():
    assert dates_in("दिनांक 14-3-26 (बैठक सं. 12) and २१/४/२६, 1.5.2.3, 9 - 10 - 2019") == ["14/3/26", "21/4/26", "9/10/2019"]
    assert dates_in("दिनांक ११.०७.२२", raw=True) == ["11.07.22"]
    assert dates_in("खसरा 64/13/2, 64/13/27, 63/10/15 और 210/1-2, 12/10/15") == ["12/10/15"]
    assert canon_date("19/09/19", this_year=2026) == (19, 9, 2019) and canon_date("1/2/98", this_year=2026) == (1, 2, 1998)
    assert canon_date("1/2/27", this_year=2026) == (1, 2, 1927) and canon_date("1/2/27", this_year=2027) == (1, 2, 2027)
    assert count_numbers("कुल 640 + २० = 720, दिनांक 12/5/25") == 4


def test_harvest_filter():
    def f(md, h):
        return harvest_filter(md, {"header": h, "footer": []}, this_year=2026)["header"]

    assert f("दिनांक 7-9-2018 ... 7-9-2018 ... 7-9-2018 को", ["8-9-2018"]) == []  # page consensus
    assert f("दिनांक 3-6-25 (बैठक सं. 18)", ["23-6-25"]) == ["23-6-25"]  # kept: the body date may be the misread one
    assert f("दिनांक 16/08/2024", ["16/8/24"]) == []  # same date, 2- vs 4-digit year
    assert f("दिनांक 05/03/2019", ["5.3.29"]) == []  # future year
    assert f("कोई तारीख नहीं", ["14-3-26"]) == ["14-3-26"]
    assert f("दिनांक 12/4/03", ["12/4/2016"]) == ["12/4/2016"]
    assert harvest_filter("दिनांक 12/4/03", {"header": ["12/4/2016"]}, this_year=2026, year_window=True)["header"] == []
