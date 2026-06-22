import re

with open('solar_flare_eda.py', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = {
    '\u2192': '->',
    '\u2190': '<-',
    '\u2013': '-',
    '\u2014': '--',
    '\u2019': "'",
    '\u201c': '"',
    '\u201d': '"',
    '\u00d7': 'x',
    '\u2212': '-',
    '\u00b1': '+/-',
    '\u00b2': '2',
    '\u03c3': 'sigma',
    '\u03b1': 'alpha',
    '\u03b2': 'beta',
    '\u2260': '!=',
    '\u2264': '<=',
    '\u2265': '>=',
    '\u00b7': '.',
    '\u00e9': 'e',
}
for uni, asc in replacements.items():
    content = content.replace(uni, asc)

with open('solar_flare_eda.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Scan for any remaining non-cp1252 chars in print lines
lines = content.split('\n')
found = []
for i, line in enumerate(lines, 1):
    try:
        line.encode('cp1252')
    except UnicodeEncodeError as e:
        found.append(f'Line {i}: {repr(line[max(0,e.start-5):e.end+5])}')

if found:
    for f in found:
        print(f)
else:
    print('All clear - no cp1252 issues found.')
