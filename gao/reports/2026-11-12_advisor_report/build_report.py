from pathlib import Path
import re, html, json
from reportlab.pdfgen import canvas
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
                               Table, TableStyle, Image, PageBreak, KeepTogether)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import letter
from PIL import Image as PILImage
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
NAME='IC多因子SNN研究阶段报告'
pdfmetrics.registerFont(TTFont('CJK','C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
pdfmetrics.registerFont(TTFont('CJKB','C:/Windows/Fonts/msyhbd.ttc',subfontIndex=0))
pdfmetrics.registerFontFamily('CJK',normal='CJK',bold='CJKB',italic='CJK',boldItalic='CJKB')
styles={
 'body':ParagraphStyle('body',fontName='CJK',fontSize=11,leading=18,spaceAfter=8,wordWrap='CJK',allowWidows=0,allowOrphans=0),
 'title':ParagraphStyle('title',fontName='CJKB',fontSize=21,leading=29,spaceAfter=16,textColor=colors.black),
 'h1':ParagraphStyle('h1',fontName='CJKB',fontSize=16,leading=24,spaceBefore=17,spaceAfter=10,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='CJKB',fontSize=12.5,leading=20,spaceBefore=12,spaceAfter=7,keepWithNext=True),
 'cell':ParagraphStyle('cell',fontName='CJK',fontSize=9.5,leading=15,wordWrap='CJK'),
 'headcell':ParagraphStyle('headcell',fontName='CJKB',fontSize=9.5,leading=15,wordWrap='CJK'),
 'caption':ParagraphStyle('caption',fontName='CJK',fontSize=9.5,leading=15,spaceAfter=10,wordWrap='CJK'),
}
def fmt(s):
    s=html.escape(s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    if s.startswith('https://'):
        s=f'<link href="{s}" color="#244d70">{s}</link>'
    return s

class Report(BaseDocTemplate):
    def __init__(self,path,**kw):
        super().__init__(str(path),pagesize=letter,leftMargin=48,rightMargin=48,topMargin=50,bottomMargin=48,
                         title='IC 股指期货多因子 SNN 与 STDP 研究阶段报告',author='研究阶段报告',**kw)
        f=Frame(self.leftMargin,self.bottomMargin,self.width,self.height,id='main',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates(PageTemplate(id='report',frames=[f],onPage=self.decorate))
    def decorate(self,c,doc):
        c.saveState();c.setFont('CJK',8);c.setFillColor(colors.HexColor('#555555'))
        c.drawString(48,768,'IC 多因子 SNN 研究   |   导师讨论稿')
        c.drawRightString(564,768,'2026-11-12')
        c.setStrokeColor(colors.HexColor('#dddddd'));c.line(48,757,564,757)
        c.drawString(48,25,'Gao 多因子适配基线与机制对照   |   历史研究结果')
        c.drawRightString(564,25,str(doc.page));c.restoreState()
    def afterFlowable(self,f):
        if isinstance(f,Paragraph) and f.style.name=='h1' and f.getPlainText()!='目录':
            title=f.getPlainText();key='section'+str(getattr(f,'section_id',0))
            self.canv.bookmarkPage(key);self.canv.addOutlineEntry(title,key,0,False)
            self.notify('TOCEntry',(0,title,self.page,key))

def main():
    source=(HERE/(NAME+'.md')).read_text(encoding='utf-8')
    lines=source.splitlines();story=[];idx=0;section=0
    while idx<len(lines):
        line=lines[idx].strip();idx+=1
        if not line:continue
        if line.startswith('# '):story.append(Paragraph(fmt(line[2:]),styles['title']));continue
        if line.startswith('## '):
            if line.startswith('## 1 '):
                story.append(PageBreak());story.append(Paragraph('目录',styles['h1']))
                toc=TableOfContents();toc.levelStyles=[ParagraphStyle('toc',fontName='CJK',fontSize=11,leading=23,spaceBefore=4,wordWrap='CJK')]
                story.append(toc);story.append(PageBreak())
            section+=1;p=Paragraph(fmt(line[3:]),styles['h1']);p.section_id=section;story.append(p);continue
        if line.startswith('### '):story.append(Paragraph(fmt(line[4:]),styles['h2']));continue
        if line.startswith('|'):
            rows=[line]
            while idx<len(lines) and lines[idx].strip().startswith('|'):
                rows.append(lines[idx].strip());idx+=1
            cells=[[x.strip() for x in r.strip('|').split('|')] for r in rows if not re.match(r'^\|[\s:|\-]+\|$',r)]
            n=len(cells[0]);widths={2:[175,341],3:[155,220,141],4:[186,110,110,110],5:[128,83,99,99,107]}.get(n,[516/n]*n)
            # Numerical comparisons use a wider descriptive first column and balanced values.
            if n==3 and any('RMSE' in x for x in cells[0]):widths=[260,128,128]
            if n==3 and ('原基线连接层' in cells[0][0] or '原基线'==cells[0][0]):widths=[200,158,158]
            if n==2 and ('文件' in cells[0][0]):widths=[255,261]
            data=[[Paragraph(fmt(x),styles['headcell' if j==0 else 'cell']) for x in row] for j,row in enumerate(cells)]
            tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT',rowSplitRange=(3,-2) if len(data)>=6 else None)
            tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef4')),
                ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#d9d9d9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
                ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8)]))
            story.extend([Spacer(1,5),tab,Spacer(1,10)]);continue
        if line.startswith('!['):
            match=re.match(r'!\[(.*?)\]\((.*?)\)',line);path=(HERE/match[2]).resolve()
            with PILImage.open(path) as im:w,h=im.size
            img=Image(str(path),width=510,height=510*h/w)
            # Keep the figure and immediate caption together on one page.
            while idx<len(lines) and not lines[idx].strip():idx+=1
            cap=lines[idx].strip();idx+=1
            story.append(KeepTogether([img,Spacer(1,6),Paragraph(fmt(cap),styles['caption'])]));continue
        # Keep each bibliography description with its following link.
        paragraph=Paragraph(fmt(line),styles['body'])
        if re.match(r'^\[R\d\]',line):paragraph.keepWithNext=True
        story.append(paragraph)
    pdf=HERE/(NAME+'.pdf');Report(pdf).multiBuild(story)
    reader=PdfReader(pdf)
    (HERE/'pdf_text_check.txt').write_text('\n\n'.join(f'PAGE {i+1}\n'+p.extract_text() for i,p in enumerate(reader.pages)),encoding='utf-8')
    (HERE/'build_manifest.json').write_text(json.dumps(dict(pages=len(reader.pages),source_characters=len(source),
        title=NAME,pdf=str(pdf),source=str(HERE/(NAME+'.md'))),ensure_ascii=False,indent=2),encoding='utf-8')
    print('pages',len(reader.pages),'source characters',len(source),'PDF',pdf)


if __name__=='__main__':main()
