"""Create an editable Word version of the reviewed report source."""
from pathlib import Path
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT=Path(__file__).resolve().parent
NAME='IC多因子SNN研究阶段报告'

def font(style,size,bold=False):
    style.font.name='Calibri';style.font.size=Pt(size);style.font.bold=bold
    style.font.color.rgb=RGBColor(0,0,0)
    style.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'微软雅黑')

def rich(p,text):
    for i,part in enumerate(re.split(r'\*\*(.*?)\*\*',text)):
        r=p.add_run(part);r.bold=bool(i%2)

def main():
    d=Document();s=d.sections[0]
    s.page_width=Inches(8.5);s.page_height=Inches(11)
    s.top_margin=s.bottom_margin=Inches(.72)
    s.left_margin=s.right_margin=Inches(.7)
    s.header_distance=s.footer_distance=Inches(.3)
    for name,size,bold in [('Normal',11,False),('Title',21,True),('Subtitle',11,False),('Heading 1',16,True),('Heading 2',12.5,True),('Caption',9.5,False)]:
        st=d.styles[name];font(st,size,bold)
        st.paragraph_format.space_after=Pt(8)
        st.paragraph_format.line_spacing=1.25
        st.paragraph_format.widow_control=True
    for name in ['Heading 1','Heading 2']:
        d.styles[name].paragraph_format.space_before=Pt(14)
        d.styles[name].paragraph_format.keep_with_next=True
    header=s.header.paragraphs[0];header.text='IC 多因子 SNN 研究  |  导师讨论稿  |  2026-11-12'
    for r in header.runs:r.font.size=Pt(8)
    foot=s.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    foot.add_run('第 ')
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');foot._p.append(field)
    foot.add_run(' 页')
    lines=(ROOT/(NAME+'.md')).read_text(encoding='utf-8').splitlines();i=0
    while i<len(lines):
        t=lines[i].strip();i+=1
        if not t:continue
        if t.startswith('# '):d.add_paragraph(t[2:],'Title');continue
        if t.startswith('## '):
            if t.startswith('## 1 '):d.add_page_break()
            d.add_paragraph(t[3:],'Heading 1');continue
        if t.startswith('### '):d.add_paragraph(t[4:],'Heading 2');continue
        if t.startswith('|'):
            raw=[t]
            while i<len(lines) and lines[i].strip().startswith('|'):
                raw.append(lines[i].strip());i+=1
            rows=[[v.strip() for v in line.strip('|').split('|')] for line in raw if not re.match(r'^\|[\s:|\-]+\|$',line)]
            n=len(rows[0]);tab=d.add_table(rows=0,cols=n);tab.alignment=WD_TABLE_ALIGNMENT.CENTER;tab.autofit=False
            widths={2:[2.45,4.65],3:[2.15,3.0,1.95],4:[2.65,1.48,1.48,1.49],5:[1.6,1.25,1.4,1.4,1.45]}[n]
            if '文件' in rows[0][0]:widths=[3.5,3.6]
            for col,w in zip(tab.columns,widths):col.width=Inches(w)
            pr=tab._tbl.tblPr;b=OxmlElement('w:tblBorders')
            for edge in ['top','left','bottom','right','insideH','insideV']:
                e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');b.append(e)
            pr.append(b)
            for j,row in enumerate(rows):
                cells=tab.add_row().cells
                trpr=tab.rows[-1]._tr.get_or_add_trPr()
                no=OxmlElement('w:cantSplit');trpr.append(no)
                if j==0:trpr.append(OxmlElement('w:tblHeader'))
                for cell,w,text in zip(cells,widths,row):
                    cell.width=Inches(w);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    cp=cell._tc.get_or_add_tcPr();m=OxmlElement('w:tcMar')
                    for side,val in [('top','90'),('bottom','90'),('left','100'),('right','100')]:
                        e=OxmlElement('w:'+side);e.set(qn('w:w'),val);e.set(qn('w:type'),'dxa');m.append(e)
                    cp.append(m)
                    if j==0:
                        sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E8EEF4');cp.append(sh)
                    p=cell.paragraphs[0];rich(p,text);p.paragraph_format.space_after=Pt(0)
                    p.paragraph_format.line_spacing=1.15
                    for r in p.runs:r.font.size=Pt(9.5);r.bold=j==0 or r.bold
            d.add_paragraph().paragraph_format.space_after=Pt(1)
            continue
        if t.startswith('!['):
            m=re.match(r'!\[(.*?)\]\((.*?)\)',t)
            p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str((ROOT/m[2]).resolve()),width=Inches(7))
            p.paragraph_format.keep_with_next=True
            continue
        p=d.add_paragraph(style='Caption' if t.startswith('图 1') else 'Normal');rich(p,t)
    d.core_properties.title='IC 股指期货多因子 SNN 与 STDP 研究阶段报告'
    d.core_properties.subject='Gao 多因子适配基线与机制诊断'
    d.core_properties.comments='由已核验的 Markdown 正文生成可编辑 Word，正文与表格均为原生文本。'
    path=ROOT/(NAME+'.docx');d.save(path)
    reopened=Document(path)
    assert len(reopened.tables)==len(d.tables)
    assert len(reopened.inline_shapes)==1
    text='\n'.join(p.text for p in reopened.paragraphs)+'\n'+'\n'.join(c.text for t in reopened.tables for row in t.rows for c in row.cells)
    for x in ['43,440','1664','0.2177','99.80%','2.153']:assert x in text,x
    print(path);print('Tables:',len(reopened.tables),'Paragraphs:',len(reopened.paragraphs),'Images:',len(reopened.inline_shapes))

if __name__=='__main__':main()
