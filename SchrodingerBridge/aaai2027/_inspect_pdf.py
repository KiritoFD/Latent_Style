from pypdf import PdfReader
r=PdfReader('paper_aaai2027.pdf')
for i,p in enumerate(r.pages):
    t=p.extract_text() or ''
    print(f'--- Page {i+1} ---')
    print(t[:1000])
    print()
