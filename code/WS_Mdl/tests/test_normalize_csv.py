import csv

from WS_Mdl.scripts.normalize_csv import normalize_csv


def test_normalize_csv_uses_minimal_quoting_and_preserves_multiline_fields(tmp_path):
    path = tmp_path / 'log.csv'
    path.write_bytes(
        (
            '"runN","model alias","Description"\n'
            '"1","NBr","Contains, a comma"\n'
            '"2","NBr","First line\nSecond line"\n'
        ).encode('utf-8')
    )

    assert normalize_csv(path)
    assert path.read_text(encoding='utf-8') == (
        'runN,model alias,Description\n'
        '1,NBr,"Contains, a comma"\n'
        '2,NBr,"First line\nSecond line"\n'
    )

    with path.open(encoding='utf-8', newline='') as file:
        assert list(csv.reader(file)) == [
            ['runN', 'model alias', 'Description'],
            ['1', 'NBr', 'Contains, a comma'],
            ['2', 'NBr', 'First line\nSecond line'],
        ]

    assert not normalize_csv(path)


def test_normalize_csv_preserves_excel_cp1252_encoding(tmp_path):
    path = tmp_path / 'log.csv'
    path.write_bytes('"runN","Description"\n"1","Wait\u2026"\n'.encode('cp1252'))

    assert normalize_csv(path)
    assert path.read_bytes() == 'runN,Description\n1,Wait\u2026\n'.encode('cp1252')
