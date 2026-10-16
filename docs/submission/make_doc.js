// Builds RyanGupta_230953288_FGAI_Assignment2.docx (submission cover document).
// Usage: node make_doc.js [summary.json]
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, ExternalHyperlink, HeadingLevel, Packer, Paragraph,
  ShadingType, Table, TableCell, TableRow, TextRun, WidthType,
} = require("docx");

const D = {
  name: "Ryan Gupta",
  reg: "230953288",
  course: "Foundations of Generative AI (ICT4403)",
  assignment: "Assignment 2 — Autoregressive Text Generation with Simple RNN, Bidirectional RNN, LSTM and GRU",
  system: "ECHO — Recurrent Text Lab: a comparative autoregressive next-word text-generation system",
  dataset: "TinyStories (Eldan & Li, 2023)",
  datasetUrl: "https://huggingface.co/datasets/roneneldan/TinyStories",
  repo: "https://github.com/RyanGupta16/ECHO",
  demo: "https://ryangupta16.github.io/ECHO/",
};
const summary = process.argv[2] && fs.existsSync(process.argv[2]) ? JSON.parse(fs.readFileSync(process.argv[2], "utf8")) : null;

const INK = "1A1A1A", MUTED = "5F6368", ACCENT = "1C5CAB", RULE = "D0D4DC";
const W = 9026; // A4 text width at 1" margins (DXA)
const LABEL_W = 3000;

const link = (url, text) =>
  new ExternalHyperlink({ link: url, children: [new TextRun({ text: text ?? url, style: "Hyperlink" })] });

const cell = (children, width, shade) =>
  new TableCell({
    width: { size: width, type: WidthType.DXA },
    margins: { top: 140, bottom: 140, left: 180, right: 180 },
    shading: shade ? { type: ShadingType.CLEAR, color: "auto", fill: shade } : undefined,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
    },
    children: Array.isArray(children) ? children : [children],
  });

const row = (label, value) =>
  new TableRow({
    children: [
      cell(new Paragraph({ children: [new TextRun({ text: label, bold: true, color: MUTED, size: 20 })] }), LABEL_W, "F4F6F9"),
      cell(new Paragraph({ children: Array.isArray(value) ? value : [new TextRun({ text: value, size: 22 })] }), W - LABEL_W),
    ],
  });

const rows = [
  row("Student name", D.name),
  row("Registration number", D.reg),
  row("Course", D.course),
  row("Assignment title", D.assignment),
  row("Title of the developed system", D.system),
  row("Dataset", [new TextRun({ text: D.dataset + " — ", size: 22 }), link(D.datasetUrl)]),
  row("GitHub repository", [link(D.repo)]),
  row("Live demo (optional)", [link(D.demo)]),
];

const body = [
  new Paragraph({ children: [new TextRun({ text: D.course.toUpperCase(), color: ACCENT, bold: true, size: 18, characterSpacing: 40 })] }),
  new Paragraph({ heading: HeadingLevel.TITLE, spacing: { before: 120, after: 80 }, children: [new TextRun({ text: "Assignment 2 Submission", bold: true, size: 48, color: INK })] }),
  new Paragraph({
    spacing: { after: 360 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: ACCENT, space: 8 } },
    children: [new TextRun({ text: "ECHO — Recurrent Text Lab", size: 28, color: MUTED })],
  }),
  new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [LABEL_W, W - LABEL_W], rows }),
];

if (summary) {
  body.push(
    new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 480, after: 160 }, children: [new TextRun({ text: "Summary of results", bold: true, size: 28, color: INK })] }),
    ...summary.paragraphs.map((p) => new Paragraph({ spacing: { after: 140 }, children: [new TextRun({ text: p, size: 22 })] })),
  );
  if (summary.table) {
    const cols = summary.table.header.length;
    const widths = Array.from({ length: cols }, (_, i) => (i === 0 ? 2426 : Math.floor((W - 2426) / (cols - 1))));
    widths[cols - 1] += W - widths.reduce((a, b) => a + b, 0);
    const mk = (cells, head) =>
      new TableRow({
        tableHeader: head,
        children: cells.map((t, i) =>
          cell(new Paragraph({ alignment: i ? AlignmentType.RIGHT : AlignmentType.LEFT, children: [new TextRun({ text: String(t), bold: head, size: 20, color: head ? MUTED : INK })] }), widths[i], head ? "F4F6F9" : undefined),
        ),
      });
    body.push(new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths, rows: [mk(summary.table.header, true), ...summary.table.rows.map((r) => mk(r, false))] }));
  }
}

const doc = new Document({
  creator: D.name,
  title: `${D.name} ${D.reg} FGAI Assignment 2`,
  styles: { default: { document: { run: { font: "Calibri", size: 22, color: INK } } } },
  sections: [{ properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } }, children: body }],
});

const out = path.join(__dirname, "RyanGupta_230953288_FGAI_Assignment2.docx");
Packer.toBuffer(doc).then((b) => {
  fs.writeFileSync(out, b);
  console.log("wrote", out);
});
