import unittest
from pathlib import Path
from bs4 import BeautifulSoup
from src.extract_book import extract_chapters, validate_book


def extract(html, **kwargs):
    return extract_chapters(BeautifulSoup(html, 'html.parser'), 'test', **kwargs)


class ExtractionTests(unittest.TestCase):
    def test_flat_multilingual_and_missing_anchors(self):
        chapters = extract('<main><h2>第一章</h2><p>Hello <em>world</em>.</p><h2>Deux</h2><p>Next</p></main>')
        self.assertEqual([c['title'] for c in chapters], ['第一章', 'Deux'])
        self.assertEqual(chapters[0]['paragraphs'][0]['text'], 'Hello world.')
        self.assertIsNone(chapters[0]['source_anchor'])
        self.assertEqual(extract('<main><h2>第一章</h2><p>Hello <em>world</em>.</p><h2>Deux</h2><p>Next</p></main>'), chapters)

    def test_nested_blocks_not_duplicated_and_inherited_classes(self):
        chapters = extract('<div class="chapter" id="ch"><h2>Start</h2><div class="comment"><p>A</p></div><blockquote><p>B</p></blockquote><ul><li>C<p>D</p></li></ul><div class="footnote"><p>E</p></div><div class="poem"><p>F</p><p>G</p></div></div>', commentary_classes={'comment'})
        paragraphs = chapters[0]['paragraphs']
        self.assertEqual([p['text'] for p in paragraphs], ['A', 'B', 'C D', 'E', 'F G'])
        self.assertEqual([p['content_type'] for p in paragraphs], ['commentary', 'body', 'body', 'footnote', 'verse'])
        self.assertEqual(chapters[0]['source_anchor'], 'ch')

    def test_filter_and_boilerplate(self):
        chapters = extract('<div id="pg-header"><p>License</p></div><h2>Preface</h2><p>P</p><h2>Chapter I</h2><p>A</p><h2>Notes</h2><p>N</p>', title_pattern=r'^Chapter')
        self.assertEqual([p['text'] for c in chapters for p in c['paragraphs']], ['A'])

    def test_custom_selectors_and_exclusions(self):
        chapters = extract('<article><h4 id="a">One</h4><div class="text">A</div><aside><div class="text">Skip</div></aside></article>', container_selector='article', heading_selector='h4', paragraph_selector='div.text', exclude_selectors=['aside'])
        self.assertEqual(chapters[0]['paragraphs'][0]['text'], 'A')
        self.assertEqual(chapters[0]['chapter_id'], 'a')

    def test_unsectioned(self):
        self.assertEqual(extract('<p>Only text</p>')[0]['title'], 'Untitled section')

    def test_duplicate_anchors_and_no_match(self):
        for html, options in [('<h2 id="x">A</h2><p>A</p><h2 id="x">B</h2><p>B</p>', {}), ('<p>A</p>', {'container_selector': '.missing'}), ('<h2>Empty</h2>', {})]:
            with self.assertRaises(ValueError):
                extract(html, **options)

    def test_validator(self):
        book = {'chapters': extract('<h2>A</h2><p>A</p>')}
        self.assertEqual(validate_book(book, 1), (1, 1))
        with self.assertRaises(ValueError):
            validate_book(book, 2)
        book['chapters'][0]['paragraphs'][0]['text'] = ' '
        with self.assertRaises(ValueError):
            validate_book(book)

    def test_local_books(self):
        for filename, count in [('the_prince.html', 26), ('art_of_war.html', 13)]:
            path = Path('data/raw') / filename
            if not path.exists():
                continue
            chapters = extract_chapters(BeautifulSoup(path.read_bytes(), 'html.parser'), filename, {'comment'}, heading_selector='h2', title_pattern=r'^chapter\s+')
            self.assertEqual(len(chapters), count)
            validate_book({'chapters': chapters}, count)
            self.assertTrue(all(c['source_anchor'] for c in chapters))


if __name__ == '__main__':
    unittest.main()
