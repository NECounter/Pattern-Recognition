using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using MathBase;
using System.Threading;


namespace TestForm
{
    public partial class Form1 : Form
    {

   
        public Form1()
        {
            InitializeComponent();
            
        }

        public Thread th1 = new Thread(new ThreadStart(()=> { }));
        private void button1_Click(object sender, EventArgs e)
        {
            //float gaussianStep = 1.6f;

            //float sigmaLimit = 1000;
            //GaussianKernel gk1 = new GaussianKernel(0);
            //Bitmap lastBit = null;

            //for (float i = 6.4f; i < sigmaLimit; i += gaussianStep)
            //{
            //    gk1.Sigma = i;
            //    Bitmap sigma1 = gk1.Conv((Bitmap)pictureBox1.Image);

            //    if (lastBit != null)
            //    {
            //        Bitmap dog = UnityTools.Diff(sigma1, lastBit);
            //        lastBit.Save(@"C:\MediaSlot\CloudDocs\Docs\课程\模式识别\Algorithms\TestImages\Records\gaussian-" + (i - gaussianStep).ToString("0.0") + ".bmp");
            //        sigma1.Save(@"C:\MediaSlot\CloudDocs\Docs\课程\模式识别\Algorithms\TestImages\Records\gaussian-" + i.ToString("0.0") + ".bmp");
            //        dog.Save(@"C:\MediaSlot\CloudDocs\Docs\课程\模式识别\Algorithms\TestImages\Records\" + i.ToString("0.0") + "-" + (i - gaussianStep).ToString("0.0") + ".bmp");
            //    }

            //    lastBit = Bitmap.FromHbitmap(sigma1.GetHbitmap());
            //}
        
            if (th1.ThreadState != ThreadState.Running)
            {
                th1 = new Thread(new ThreadStart(() =>
                {
                    SIFTSolver gk = new SIFTSolver(1.6f);
                    gk.ColorFormat = ColorFormat.GrayScale;
                    Bitmap[,] pyramid = gk.GetGaussianPyramid(UnityTools.ConvertToGray((Bitmap)pictureBox1.Image), 1, 3);
                    //Bitmap[,] pyramid = gk.GetGaussianPyramid((Bitmap)pictureBox1.Image, 3, 5);
                    Bitmap[,] dogs = gk.GetDoGs(pyramid);

                    for (int i = 0; i < pyramid.GetLength(0); i++)
                    {
                        for (int j = 0; j < pyramid.GetLength(1); j++)
                        {
                            pyramid[i, j].Save(@"C:\MediaSlot\CloudDocs\Docs\课程\模式识别\Algorithms\TestImages\Records\" + i + "-" + j + ".bmp");
                        }
                    }

                    for (int i = 0; i < dogs.GetLength(0); i++)
                    {
                        for (int j = 0; j < dogs.GetLength(1); j++)
                        {
                            dogs[i, j].Save(@"C:\MediaSlot\CloudDocs\Docs\课程\模式识别\Algorithms\TestImages\Records\dogs-" + i + "-" + j + ".bmp");
                        }
                    }
                }));

                th1.Start();
            }

            
         

            

           
       

            

            










        }


        private void Form1_Load(object sender, EventArgs e)
        {
            pictureBox1.Image = Image.FromFile(@"C:\MediaSlot\CloudDocs\Docs\课程\模式识别\Algorithms\TestImages\1.bmp");
        }

        private void button2_Click(object sender, EventArgs e)
        {
            textBox2.Text = "";
            SIFTSolver gk1 = new SIFTSolver(Convert.ToSingle(textBox1.Text));
            pictureBox2.Image = gk1.Conv((Bitmap)pictureBox1.Image);
            int[,] table = gk1.GetGaussianTable();

            for (int i = 0; i < table.GetLength(1); i++)
            {
                for (int j = 0; j < table.GetLength(0); j++)
                {

                    textBox2.AppendText(table[j, i].ToString("0000000") + " ");
                }
                textBox2.AppendText("\r\n");
            }

        }

        private void button3_Click(object sender, EventArgs e)
        {
            pictureBox2.Image = UnityTools.ConvertToGray((Bitmap)pictureBox1.Image);
        }
    }
}
