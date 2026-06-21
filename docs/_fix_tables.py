"""Konwertuje longtable (pandoc) na czytelne tabele z zebra-stripingiem.

Strategia hybrydowa:
- Tabele waskie/liczbowe (do 7 kolumn, najwyzej jedna kolumna tekstowa z lewej)
  -> tabularx{\\linewidth}: rozciagniete na pelna szerokosc, kolumny liczbowe do prawej.
- Tabele szerokie (8+ kolumn lub z dodatkowa kolumna tekstowa, np. Status/Typ)
  -> \\resizebox{\\linewidth}: skalowane w calosci, bez lamania pojedynczych slow.
Obie z zebra-stripingiem (co drugi wiersz delikatnie szary)."""
import re

src = open("RAPORT_V2.tex").read()

def convert(block):
    m = re.match(r"\\begin\{longtable\}\[\]\{(@\{\}[^}]*@\{\})\}(.*)\\end\{longtable\}", block, re.S)
    if not m:
        return block
    colspec_raw, body = m.group(1), m.group(2)
    cols = re.sub(r"@\{\}", "", colspec_raw)
    # usun longtable-specyficzne komendy
    for cmd in [r"\endhead", r"\endfirsthead", r"\endlastfoot"]:
        body = body.replace(cmd, "")
    body = re.sub(r"\\noalign\{\}", "", body).strip()

    n = len(cols)
    # liczba kolumn tekstowych (l/c) poza pierwsza -> sygnal "szerokiej" tabeli z tekstem
    text_cols_after_first = sum(1 for c in cols[1:] if c == "l")
    wide = (n >= 8) or (text_cols_after_first >= 1)

    if wide:
        # resizebox: skaluje cala tabele, nie lamie slow; tabular zwykly
        return (
            "\\begingroup\n\\rowcolors{2}{zebragray}{white}\n"
            "\\noindent\\resizebox{\\linewidth}{!}{%\n"
            f"\\begin{{tabular}}{{{cols}}}\n{body}\n\\end{{tabular}}%\n}}\n"
            "\\endgroup\n"
        )
    else:
        # tabularx: pierwsza kolumna L (tekst), reszta R (liczby do prawej)
        new_cols = "".join("L" if i == 0 else ("R" if c in "rc" else "L")
                           for i, c in enumerate(cols))
        return (
            "\\begingroup\n\\rowcolors{2}{zebragray}{white}\n"
            f"\\begin{{tabularx}}{{\\linewidth}}{{{new_cols}}}\n{body}\n\\end{{tabularx}}\n"
            "\\endgroup\n"
        )

pattern = re.compile(r"\\begin\{longtable\}\[\]\{@\{\}[^}]*@\{\}\}.*?\\end\{longtable\}", re.S)
out = pattern.sub(lambda mm: convert(mm.group(0)), src)
open("RAPORT_V2.tex", "w").write(out)
print(f"przekonwertowano {len(pattern.findall(src))} tabel (hybryda tabularx/resizebox + zebra)")
