using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SCDLabel
{
    public partial class FormMain : Form
    {
        public FormMain()
        {
            InitializeComponent();
        }

        float zoom = 1;
        string inputDirectory = "";
        string outputDirectory = "";

        string fileName = "";
        PictureBox picture;

        private unsafe void listBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(inputDirectory)) return;
            if (string.IsNullOrEmpty(outputDirectory)) return;

            selectedMarker = null;
            this.fileName = listBox1.SelectedItem.ToString();

            if (picture != null)
                picture.Dispose();

            picture = new PictureBox();
            panel2.Controls.Add(picture);
            picture.SizeMode = PictureBoxSizeMode.StretchImage;
            picture.Location = new Point(0, 0);

            Bitmap bmp = new Bitmap(inputDirectory + "\\" + fileName);

            // edit the saturation of bmp.

            BitmapData bmpData = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height),
                ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);

            byte* start = (byte*)bmpData.Scan0;
            int offset = bmpData.Stride - 3 * bmp.Width;
            for (int y = 0; y < bmp.Height; y++) {
                for (int x = 0; x < bmp.Width; x++) {

                    var hsv = RgbToHsv((start[2], start[1], start[0]));
                    var rgb = HsvToRgb((hsv.h, Math.Min(255, Convert.ToInt32( Convert.ToInt32(numericUpDown2.Value) * hsv.s / 100f)), hsv.v));

                    start[2] = (byte)rgb.r;
                    start[1] = (byte)rgb.g;
                    start[0] = (byte)rgb.b;

                    start += 3;
                }
                start += offset;
            }

            bmp.UnlockBits(bmpData);

            picture.Size = new Size(Convert.ToInt32(bmp.Width * zoom), Convert.ToInt32(bmp.Height * zoom));
            picture.Image = bmp;

            picture.Paint += Picture_Paint;
            picture.MouseMove += Picture_MouseMove;
            picture.MouseDown += Picture_MouseDown;
            picture.MouseUp += Picture_MouseUp;

            markers.Clear();

            if (File.Exists(outputDirectory + "\\" + fileName.Replace("." + fileName.Split('.').Last(), ".txt"))) {

                FileStream fs = new FileStream(outputDirectory + "\\" + fileName.Replace("." + fileName.Split('.').Last(), ".txt"), FileMode.OpenOrCreate);
                StreamReader reader = new StreamReader(fs);
                string str = reader.ReadToEnd();

                foreach (var item in str.Split('\n')) {
                    if(string.IsNullOrEmpty(item.Replace("\r","").Replace("\n",""))) continue;
                    markers.Add(new Marker(item.Replace("\r", "").Replace("\n", "")));
                }

                reader.Close();
                fs.Close();
            }
        }

        public static (int h, int s, int v) RgbToHsv((int r, int g, int b) rgb)
        {
            float min, max, tmp, H, S, V;
            float R = rgb.r / 255f, G = rgb.g / 255f, B = rgb.b / 255f;
            tmp = Math.Min(R, G);
            min = Math.Min(tmp, B);
            tmp = Math.Max(R, G);
            max = Math.Max(tmp, B);

            H = 0;
            if (max == min) {
                H = 0;
            } else if (max == R && G > B) {
                H = 60 * (G - B) * 1.0f / (max - min) + 0;
            } else if (max == R && G < B) {
                H = 60 * (G - B) * 1.0f / (max - min) + 360;
            } else if (max == G) {
                H = H = 60 * (B - R) * 1.0f / (max - min) + 120;
            } else if (max == B) {
                H = H = 60 * (R - G) * 1.0f / (max - min) + 240;
            }

            if (max == 0) {
                S = 0;
            } else {
                S = (max - min) * 1.0f / max;
            }

            V = max;
            return ((int)H, (int)(S * 255), (int)(V * 255));
        }

        public static (int r, int g, int b) HsvToRgb((int h, int s, int v) hsv)
        {
            if (hsv.h == 360) hsv.h = 359;
            float R = 0f, G = 0f, B = 0f;
            if (hsv.s == 0) {
                return (hsv.v, hsv.v, hsv.v);
            }
            float S = hsv.s * 1.0f / 255, V = hsv.v * 1.0f / 255;
            int H1 = (int)(hsv.h * 1.0f / 60), H = hsv.h;
            float F = H * 1.0f / 60 - H1;
            float P = V * (1.0f - S);
            float Q = V * (1.0f - F * S);
            float T = V * (1.0f - (1.0f - F) * S);
            switch (H1) {
                case 0: R = V; G = T; B = P; break;
                case 1: R = Q; G = V; B = P; break;
                case 2: R = P; G = V; B = T; break;
                case 3: R = P; G = Q; B = V; break;
                case 4: R = T; G = P; B = V; break;
                case 5: R = V; G = P; B = Q; break;
            }
            R = R * 255;
            G = G * 255;
            B = B * 255;
            while (R > 255) R -= 255;
            while (R < 0) R += 255;
            while (G > 255) G -= 255;
            while (G < 0) G += 255;
            while (B > 255) B -= 255;
            while (B < 0) B += 255;
            return ((int)R, (int)G, (int)B);
        }

        Stage drawStage = Stage.Idle;

        enum Stage
        {
            Directioning,
            Sizing,
            Disperation,
            Idle
        }

        PointF point1 = new Point(0, 0);
        PointF point2 = new Point(0, 0);
        float size;
        float disperation;

        private void Picture_MouseUp(object sender, MouseEventArgs e)
        {
            PointF currentPoint = new PointF(e.X / zoom, e.Y / zoom);
            PointF center = new PointF((point1.X + point2.X) / 2, (point1.Y + point2.Y) / 2);
            switch (this.drawStage) {
                case Stage.Directioning:
                    this.point2 = currentPoint;

                    if (Marker.GetDistance(point1, point2) <= 1e-2)
                        this.drawStage = Stage.Idle;

                    this.drawStage = Stage.Sizing;
                    break;
                case Stage.Sizing:
                    break;
                case Stage.Disperation:
                    if (size > 1e-2 && disperation > 1e-2)
                        this.markers.Add(new Marker(
                            point2, point1, size, disperation));
                    this.drawStage = Stage.Idle;
                    break;
                case Stage.Idle:
                    break;
                default:
                    throw new NotImplementedException();
            }
        }

        Marker selectedMarker;

        private void Picture_MouseDown(object sender, MouseEventArgs e)
        {
            PointF currentPoint = new PointF(e.X / zoom, e.Y / zoom);
            PointF center = new PointF((point1.X + point2.X) / 2, (point1.Y + point2.Y) / 2);
            switch (this.drawStage) {
                case Stage.Sizing:
                    size = 2* Marker.GetLength(point1, point2, currentPoint);
                    this.drawStage = Stage.Disperation;
                    break;
                case Stage.Disperation:
                    disperation = Marker.GetDistance(center, currentPoint);
                    break;
                case Stage.Directioning:
                    throw new NotImplementedException();
                case Stage.Idle:

                    foreach (var item in markers) {
                        if (item.Inside(currentPoint)) {
                            selectedMarker = item;
                            return;
                        }
                    }

                    this.drawStage = Stage.Directioning;
                    this.point1 = currentPoint;
                    break;
                default:
                    throw new NotImplementedException();
            }
        }

        PointF currentPointMv;
        PointF centerMv;
        private void Picture_MouseMove(object sender, MouseEventArgs e)
        {
            currentPointMv = new PointF(e.X / zoom, e.Y / zoom);
            centerMv = new PointF((point1.X + point2.X) / 2, (point1.Y + point2.Y) / 2);
            (sender as PictureBox).Refresh();
        }

        private void Picture_Paint(object sender, PaintEventArgs e)
        {
            foreach (var item in markers) {
                if (selectedMarker == item)
                    item.DrawSelected(e.Graphics, zoom);
                else
                    item.Draw(e.Graphics, zoom);
            }

            var g = e.Graphics;
            Pen whitePen = new Pen(Brushes.White, 2);
            Pen highlightPen = new Pen(SystemBrushes.Highlight, 2);

            PointF center = new PointF((point1.X + point2.X) / 2, (point1.Y + point2.Y) / 2);
            switch (this.drawStage) {
                case Stage.Directioning:
                    g.DrawLine(highlightPen,
                    new PointF(point1.X * zoom, point1.Y * zoom),
                    new PointF(currentPointMv.X * zoom, currentPointMv.Y * zoom));

                    g.FillRectangle(Brushes.White, new Rectangle(
                    Convert.ToInt32(point1.X * zoom) - 5, Convert.ToInt32(point1.Y * zoom) - 5, 10, 10));
                    g.DrawRectangle(SystemPens.Highlight, new Rectangle(
                        Convert.ToInt32(point1.X * zoom) - 5, Convert.ToInt32(point1.Y * zoom) - 5, 10, 10));

                    g.FillRectangle(Brushes.White, new Rectangle(
                        Convert.ToInt32(currentPointMv.X * zoom) - 5, Convert.ToInt32(currentPointMv.Y * zoom) - 5, 10, 10));
                    g.DrawRectangle(SystemPens.Highlight, new Rectangle(
                        Convert.ToInt32(currentPointMv.X * zoom) - 5, Convert.ToInt32(currentPointMv.Y * zoom) - 5, 10, 10));

                    g.FillEllipse(Brushes.White, new Rectangle(
                        Convert.ToInt32((point1.X + currentPointMv.X) * zoom / 2.0f) - 5, Convert.ToInt32((point1.Y + currentPointMv.Y) * zoom / 2.0f) - 5, 10, 10));
                    g.DrawEllipse(SystemPens.Highlight, new Rectangle(
                        Convert.ToInt32((point1.X + currentPointMv.X) * zoom / 2.0f) - 5, Convert.ToInt32((point1.Y + currentPointMv.Y) * zoom / 2.0f) - 5, 10, 10));
                    break;
                case Stage.Sizing:
                    g.DrawLine(highlightPen,
                    new PointF(point1.X * zoom, point1.Y * zoom),
                    new PointF(point2.X * zoom, point2.Y * zoom));

                    g.FillRectangle(Brushes.White, new Rectangle(
                    Convert.ToInt32(point1.X * zoom) - 5, Convert.ToInt32(point1.Y * zoom) - 5, 10, 10));
                    g.DrawRectangle(SystemPens.Highlight, new Rectangle(
                        Convert.ToInt32(point1.X * zoom) - 5, Convert.ToInt32(point1.Y * zoom) - 5, 10, 10));

                    g.FillRectangle(Brushes.White, new Rectangle(
                        Convert.ToInt32(point2.X * zoom) - 5, Convert.ToInt32(point2.Y * zoom) - 5, 10, 10));
                    g.DrawRectangle(SystemPens.Highlight, new Rectangle(
                        Convert.ToInt32(point2.X * zoom) - 5, Convert.ToInt32(point2.Y * zoom) - 5, 10, 10));

                    g.FillEllipse(Brushes.White, new Rectangle(
                        Convert.ToInt32((point1.X + point2.X) * zoom / 2.0f) - 5, Convert.ToInt32((point1.Y + point2.Y) * zoom / 2.0f) - 5, 10, 10));
                    g.DrawEllipse(SystemPens.Highlight, new Rectangle(
                        Convert.ToInt32((point1.X + point2.X) * zoom / 2.0f) - 5, Convert.ToInt32((point1.Y + point2.Y) * zoom / 2.0f) - 5, 10, 10));

                    size = 2* Marker.GetLength(point1, point2, currentPointMv);
                    g.TranslateTransform(center.X * zoom, center.Y * zoom);
                    g.RotateTransform(Marker.GetDegree(point1, point2));

                    float w = Marker.GetDistance(point1, point2);
                    float h = size;

                    g.DrawRectangle(highlightPen,
                        new Rectangle(-Convert.ToInt32(w / 2 * zoom), -Convert.ToInt32(h / 2 * zoom),
                        Convert.ToInt32(w * zoom), Convert.ToInt32(h * zoom)));

                    g.RotateTransform(-Marker.GetDegree(point1, point2));
                    g.TranslateTransform(-center.X * zoom, -center.Y * zoom);

                    break;
                case Stage.Disperation:
                    disperation = Marker.GetDistance(center, currentPointMv);
                    var temp = new Marker(
                        point2, point1, size, disperation);
                    temp.Draw(g, zoom);
                    break;
                case Stage.Idle:
                    foreach (var item in markers) {
                        if (item.Inside(currentPointMv)) {
                            item.DrawEmphasis(g, zoom);
                        }
                    }
                    break;
                default:
                    throw new NotImplementedException();
            }
        }

        List<Marker> markers = new List<Marker>() {
            new Marker(new PointF(10, 10), new PointF(20,20), 10, 30) };

        private void toolStripButton3_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog folderBrowserDialog = new FolderBrowserDialog();
            if (folderBrowserDialog.ShowDialog() == DialogResult.OK) {
                this.inputDirectory = folderBrowserDialog.SelectedPath;
            }

            this.listBox1.Items.Clear();
            if (string.IsNullOrEmpty(this.inputDirectory)) return;

            DirectoryInfo info = new DirectoryInfo(this.inputDirectory);
            foreach (var item in info.GetFiles()) { 
                listBox1.Items.Add(item.Name);
            }
        }

        private void toolStripButton5_Click(object sender, EventArgs e)
        {
            FolderBrowserDialog folderBrowserDialog = new FolderBrowserDialog();
            if (folderBrowserDialog.ShowDialog() == DialogResult.OK) {
                this.outputDirectory = folderBrowserDialog.SelectedPath;
            }
        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            this.zoom *= 2;
            if (picture == null) return;
            if (picture.Image == null) return;

            picture.Size = new Size(Convert.ToInt32(picture.Image.Width * zoom), Convert.ToInt32(picture.Image.Height * zoom));
            picture.Refresh();
        }

        public class Marker
        {
            public Marker(PointF tail, PointF head, float width, float disperation)
            {
                this.Tail = tail;
                this.Head = head;
                this.Width = width;
                this.Disperation = disperation;
            }

            public Marker(string expr)
            {
                string[] spls = expr.Split(';');
                this.Tail = new PointF(float.Parse(spls[0]), float.Parse(spls[1]));
                this.Head = new PointF(float.Parse(spls[2]), float.Parse(spls[3]));
                this.Width = float.Parse(spls[4]);
                this.Disperation = float.Parse(spls[5]);
            }

            public override string ToString()
            {
                return Tail.X.ToString("F2") + ";" + 
                    Tail.Y.ToString("F2") + ";" + 
                    Head.X.ToString("F2") + ";" + 
                    Head.Y.ToString("F2") + ";" + 
                    Width.ToString("F2") + ";" + 
                    Disperation.ToString("F2");
            }

            PointF Tail;
            PointF Head;
            float Width;
            float Disperation;

            public void ChangeTailHead()
            {
                float hx = Head.X, hy = Head.Y;
                Head.X = Tail.X;
                Head.Y = Tail.Y;
                Tail.X = hx;
                Tail.Y = hy;
            }

            public void Draw(Graphics g, float zoom)
            {
                Pen whitePen = new Pen(Brushes.White, 2);
                Pen highlightPen = new Pen(SystemBrushes.Highlight, 2);

                Point center = new Point(Convert.ToInt32((Head.X + Tail.X) * zoom / 2.0f), Convert.ToInt32((Head.Y + Tail.Y) * zoom / 2.0f));
                g.TranslateTransform(center.X, center.Y);
                g.RotateTransform(GetDegree(this.Head, this.Tail));

                float w = GetDistance(this.Head, this.Tail);
                float h = Width;

                g.DrawRectangle(highlightPen,
                    new Rectangle(-Convert.ToInt32(w / 2 * zoom), -Convert.ToInt32(h / 2 * zoom),
                    Convert.ToInt32(w * zoom), Convert.ToInt32(h * zoom)));

                g.RotateTransform(-GetDegree(this.Head, this.Tail));
                g.TranslateTransform(-center.X, -center.Y);

                g.DrawLine(highlightPen,
                    new PointF(Head.X * zoom, Head.Y * zoom),
                    new PointF(Tail.X * zoom, Tail.Y * zoom));

                g.FillRectangle(Brushes.Green, new Rectangle(
                    Convert.ToInt32(Head.X * zoom) - 5, Convert.ToInt32(Head.Y * zoom) - 5, 10, 10));
                g.DrawRectangle(SystemPens.Highlight, new Rectangle(
                    Convert.ToInt32(Head.X * zoom) - 5, Convert.ToInt32(Head.Y * zoom) - 5, 10, 10));

                g.FillRectangle(Brushes.Yellow, new Rectangle(
                    Convert.ToInt32(Tail.X * zoom) - 5, Convert.ToInt32(Tail.Y * zoom) - 5, 10, 10));
                g.DrawRectangle(SystemPens.Highlight, new Rectangle(
                    Convert.ToInt32(Tail.X * zoom) - 5, Convert.ToInt32(Tail.Y * zoom) - 5, 10, 10));

                g.FillEllipse(Brushes.White, new Rectangle(
                    Convert.ToInt32((Head.X + Tail.X) * zoom / 2.0f) - 5, Convert.ToInt32((Head.Y + Tail.Y) * zoom / 2.0f) - 5, 10, 10));
                g.DrawEllipse(SystemPens.Highlight, new Rectangle(
                    Convert.ToInt32((Head.X + Tail.X) * zoom / 2.0f) - 5, Convert.ToInt32((Head.Y + Tail.Y) * zoom / 2.0f) - 5, 10, 10));

                g.DrawEllipse(highlightPen, new Rectangle(
                    Convert.ToInt32((Head.X + Tail.X) * zoom / 2.0f - Disperation * zoom), 
                    Convert.ToInt32((Head.Y + Tail.Y) * zoom / 2.0f - Disperation * zoom),
                    Convert.ToInt32(Disperation * zoom * 2),
                    Convert.ToInt32(Disperation * zoom * 2)));
            }

            public void DrawEmphasis(Graphics g, float zoom)
            {
                Pen whitePen = new Pen(Brushes.White, 2);
                Pen highlightPen = new Pen(Brushes.Red, 2);

                Point center = new Point(Convert.ToInt32((Head.X + Tail.X) * zoom / 2.0f), Convert.ToInt32((Head.Y + Tail.Y) * zoom / 2.0f));
                g.TranslateTransform(center.X, center.Y);
                g.RotateTransform(GetDegree(this.Head, this.Tail));

                float w = GetDistance(this.Head, this.Tail);
                float h = Width;

                g.DrawRectangle(highlightPen,
                    new Rectangle(-Convert.ToInt32(w / 2 * zoom), -Convert.ToInt32(h / 2 * zoom),
                    Convert.ToInt32(w * zoom), Convert.ToInt32(h * zoom)));

                g.RotateTransform(-GetDegree(this.Head, this.Tail));
                g.TranslateTransform(-center.X, -center.Y);

                g.DrawLine(highlightPen,
                    new PointF(Head.X * zoom, Head.Y * zoom),
                    new PointF(Tail.X * zoom, Tail.Y * zoom));

                g.FillRectangle(Brushes.White, new Rectangle(
                    Convert.ToInt32(Head.X * zoom) - 5, Convert.ToInt32(Head.Y * zoom) - 5, 10, 10));
                g.DrawRectangle(SystemPens.Highlight, new Rectangle(
                    Convert.ToInt32(Head.X * zoom) - 5, Convert.ToInt32(Head.Y * zoom) - 5, 10, 10));

                g.FillRectangle(Brushes.White, new Rectangle(
                    Convert.ToInt32(Tail.X * zoom) - 5, Convert.ToInt32(Tail.Y * zoom) - 5, 10, 10));
                g.DrawRectangle(SystemPens.Highlight, new Rectangle(
                    Convert.ToInt32(Tail.X * zoom) - 5, Convert.ToInt32(Tail.Y * zoom) - 5, 10, 10));

                g.FillEllipse(Brushes.White, new Rectangle(
                    Convert.ToInt32((Head.X + Tail.X) * zoom / 2.0f) - 5, Convert.ToInt32((Head.Y + Tail.Y) * zoom / 2.0f) - 5, 10, 10));
                g.DrawEllipse(SystemPens.Highlight, new Rectangle(
                    Convert.ToInt32((Head.X + Tail.X) * zoom / 2.0f) - 5, Convert.ToInt32((Head.Y + Tail.Y) * zoom / 2.0f) - 5, 10, 10));

                g.DrawEllipse(highlightPen, new Rectangle(
                    Convert.ToInt32((Head.X + Tail.X) * zoom / 2.0f - Disperation * zoom),
                    Convert.ToInt32((Head.Y + Tail.Y) * zoom / 2.0f - Disperation * zoom),
                    Convert.ToInt32(Disperation * zoom * 2),
                    Convert.ToInt32(Disperation * zoom * 2)));
            }

            public void DrawSelected(Graphics g, float zoom)
            {
                Pen whitePen = new Pen(Brushes.Red, 2);
                Pen highlightPen = new Pen(Brushes.Red, 2);

                Point center = new Point(Convert.ToInt32((Head.X + Tail.X) * zoom / 2.0f), Convert.ToInt32((Head.Y + Tail.Y) * zoom / 2.0f));
                g.TranslateTransform(center.X, center.Y);
                g.RotateTransform(GetDegree(this.Head, this.Tail));

                float w = GetDistance(this.Head, this.Tail);
                float h = Width;

                g.DrawRectangle(highlightPen,
                    new Rectangle(-Convert.ToInt32(w / 2 * zoom), -Convert.ToInt32(h / 2 * zoom),
                    Convert.ToInt32(w * zoom), Convert.ToInt32(h * zoom)));

                g.RotateTransform(-GetDegree(this.Head, this.Tail));
                g.TranslateTransform(-center.X, -center.Y);

                g.DrawLine(highlightPen,
                    new PointF(Head.X * zoom, Head.Y * zoom),
                    new PointF(Tail.X * zoom, Tail.Y * zoom));

                g.FillRectangle(Brushes.White, new Rectangle(
                    Convert.ToInt32(Head.X * zoom) - 5, Convert.ToInt32(Head.Y * zoom) - 5, 10, 10));
                g.DrawRectangle(Pens.Red, new Rectangle(
                    Convert.ToInt32(Head.X * zoom) - 5, Convert.ToInt32(Head.Y * zoom) - 5, 10, 10));

                g.FillRectangle(Brushes.White, new Rectangle(
                    Convert.ToInt32(Tail.X * zoom) - 5, Convert.ToInt32(Tail.Y * zoom) - 5, 10, 10));
                g.DrawRectangle(Pens.Red, new Rectangle(
                    Convert.ToInt32(Tail.X * zoom) - 5, Convert.ToInt32(Tail.Y * zoom) - 5, 10, 10));

                g.FillEllipse(Brushes.White, new Rectangle(
                    Convert.ToInt32((Head.X + Tail.X) * zoom / 2.0f) - 5, Convert.ToInt32((Head.Y + Tail.Y) * zoom / 2.0f) - 5, 10, 10));
                g.DrawEllipse(Pens.Red, new Rectangle(
                    Convert.ToInt32((Head.X + Tail.X) * zoom / 2.0f) - 5, Convert.ToInt32((Head.Y + Tail.Y) * zoom / 2.0f) - 5, 10, 10));

                g.DrawEllipse(highlightPen, new Rectangle(
                    Convert.ToInt32((Head.X + Tail.X) * zoom / 2.0f - Disperation * zoom),
                    Convert.ToInt32((Head.Y + Tail.Y) * zoom / 2.0f - Disperation * zoom),
                    Convert.ToInt32(Disperation * zoom * 2),
                    Convert.ToInt32(Disperation * zoom * 2)));
            }

            public bool Inside(PointF picture)
            {
                var rect = new RectangleF(
                    (Head.X + Tail.X) / 2.0f - Disperation,
                    (Head.Y + Tail.Y) / 2.0f - Disperation,
                    2 * Disperation,
                    2 * Disperation);

                GraphicsPath path = new GraphicsPath();
                path.AddEllipse(rect);
                path.CloseFigure();
                return path.IsVisible(picture);
            }

            public static float GetLength(PointF head, PointF tail, PointF current)
            {
                float b = -(tail.X - head.X);
                float a = tail.Y - head.Y;
                float c0 = -a * tail.X - b * tail.Y;
                float c1 = -a * current.X - b * current.Y;

                if (a * a + b * b <= 0e-2) return 0;

                return Convert.ToSingle( Math.Abs(c0 - c1) / Math.Sqrt(a * a + b * b) );
            }

            public static float GetDegree(PointF head, PointF tail)
            {
                float deltaX = tail.X - head.X;
                float deltaY = tail.Y - head.Y;

                if(Math.Abs( deltaX) < 1e-2) return 90;
                float tan = deltaY / deltaX;

                return Convert.ToSingle(Math.Atan(tan) * 180 / Math.PI);
            }

            public static float GetDistance(PointF head, PointF tail)
            {
                float deltaX = tail.X - head.X;
                float deltaY = tail.Y - head.Y;

                return Convert.ToSingle(Math.Sqrt(Math.Pow(deltaX, 2) + Math.Pow(deltaY, 2)));
            }
        }

        private void toolStripButton2_Click(object sender, EventArgs e)
        {
            this.zoom /= 2;
            if (picture == null) return;
            if (picture.Image == null) return;

            picture.Size = new Size(Convert.ToInt32(picture.Image.Width * zoom), Convert.ToInt32(picture.Image.Height * zoom));
            picture.Refresh();
        }

        private void toolStripButton3_Click_1(object sender, EventArgs e)
        {
            if(selectedMarker!=null)
                this.markers.Remove(selectedMarker);
            this.Refresh();
        }

        private void toolStripButton4_Click(object sender, EventArgs e)
        {
            if (File.Exists(outputDirectory + "\\" + fileName.Replace("." + fileName.Split('.').Last(), ".txt")))
                File.Delete(outputDirectory + "\\" + fileName.Replace("." + fileName.Split('.').Last(), ".txt"));

            FileStream fs = new FileStream(outputDirectory + "\\" + fileName.Replace("." + fileName.Split('.').Last(), ".txt"), FileMode.OpenOrCreate);
            StreamWriter writer = new StreamWriter(fs);
            string str = "";

            foreach (var item in markers) {
                str = str + item.ToString() + "\n";
            }

            writer.WriteLine(str);
            writer.Flush();
            writer.Close();
            fs.Close();
        }

        private void toolStripButton7_Click(object sender, EventArgs e)
        {
            if (selectedMarker != null)
                this.selectedMarker.ChangeTailHead();
            else {
                foreach (var item in markers) {
                    item.ChangeTailHead();
                }
            }

            Refresh();
            toolStripButton4_Click(null, null);
        }
    }
}
