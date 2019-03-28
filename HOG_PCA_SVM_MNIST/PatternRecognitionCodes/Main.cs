/*
 * Created By Qiao 
 * 2019.1
*/
using Emgu.CV;
using Emgu.CV.ML;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using Accord.Statistics;
using Accord.Statistics.Analysis;
using Accord.Statistics.Models.Regression.Linear;
using System.Threading;

namespace PatternRecognitionCodes
{
    public partial class Form1 : Form
    {
        #region Global Variables

        /// <summary>
        /// 画图，初始点X坐标
        /// </summary>
        private int lineStartX = 0;
        /// <summary>
        /// 画图，初始点Y坐标
        /// </summary>
        private int lineStartY = 0;
        /// <summary>
        /// 是否在画图
        /// </summary>
        private bool drawkine = false;
        /// <summary>
        /// GDI图像
        /// </summary>
        private Graphics g;
        /// <summary>
        /// 测试用的SVM实例
        /// </summary>
        private SVM SVMTester;
        private bool isImage = false;
        #endregion


        #region From Events
        public Form1()
        {
            InitializeComponent();
        }

        /// <summary>
        /// 窗体载入
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Form1_Load(object sender, EventArgs e)
        {
            //新建画布
            pictureBox1.Image = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            g = Graphics.FromImage(pictureBox1.Image);
            g.Clear(Color.White);
            //实例化SVM，载入支持向量参数
            SVMTester = new SVM();
            SVMTester.Load(@"..\..\MNIST\SVMTrainResult_HOG_PCA.xml");     
        }
        /// <summary>
        /// 将原数据进行HOG特征提取
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void BT_HOG(object sender, EventArgs e)
        {
            //实例化HOG解释器 OpenCV
            HOGDescriptor hog = new HOGDescriptor(new Size(28, 28), //图像大小
                                                  new Size(14, 14), //block大小
                                                  new Size(7, 7), //block stride
                                                  new Size(7, 7), //cell大小
                                                                9); //特征维度
            //读取数据
            List<double[]> inputDataTrain = ReadFileToList(@"..\..\MNIST\mnist_train_small.csv", ',');
            List<double[]> inputDataTest = ReadFileToList(@"..\..\MNIST\mnist_test.csv", ',');
            //创建文件写入器，把HOG后的数据写入文件系统
            //StreamWriter streamWriter1 = new StreamWriter(@"..\..\MNIST\mnist_train_small_HOG.csv");
            //StreamWriter streamWriter2 = new StreamWriter(@"..\..\MNIST\mnist_test_HOG.csv");

            //提取测试集的HOG特征
            for (int j = 0; j < inputDataTrain.Count; j++)
            {
                string lineText = inputDataTrain[j][0].ToString();
                //先转化为Bitmap
                Bitmap tempBitmap = new Bitmap(28, 28);
                for (int i = 1; i < inputDataTrain[0].Length; i++)
                {
                    tempBitmap.SetPixel((i - 1) % 28, (i - 1) / 28,
                        Color.FromArgb(255, (int)inputDataTrain[j][i], (int)inputDataTrain[j][i], (int)inputDataTrain[j][i]));
                }
                Image<Gray, byte> image = new Image<Gray, byte>(tempBitmap);
                //提取Bitmap的HOG特征
                float[] hogVector = hog.Compute(image);

                //将特征序列化后写入文件系统
                foreach (var item in hogVector)
                {
                    lineText += "," + item.ToString();
                }
                //streamWriter1.WriteLine(lineText);
            }
            //streamWriter1.Close();

            //提取训练集的HOG特征
            for (int j = 0; j < inputDataTest.Count; j++)
            {
                string lineText = inputDataTest[j][0].ToString();
                Bitmap tempBitmap = new Bitmap(28, 28);
                for (int i = 1; i < inputDataTest[0].Length; i++)
                {
                    tempBitmap.SetPixel((i - 1) % 28, (i - 1) / 28,
                        Color.FromArgb(255, (int)inputDataTest[j][i], (int)inputDataTest[j][i], (int)inputDataTest[j][i]));
                }
                Image<Gray, byte> image = new Image<Gray, byte>(tempBitmap);
                float[] hogVector = hog.Compute(image);

                foreach (var item in hogVector)
                {
                    lineText += "," + item.ToString();
                }

                //streamWriter2.WriteLine(lineText);
            }
            //streamWriter2.Close();
        }

        /// <summary>
        /// 对原数据进行PCA操作
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void BT_PCA(object sender, EventArgs e)
        {
            //读取数据
            List<double[]> inputDataTrain = ReadFileToList(@"..\..\MNIST\mnist_train_small_HOG.csv", ',');
            List<double[]> inputDataTest = ReadFileToList(@"..\..\MNIST\mnist_test_HOG.csv", ',');

            //创建文件写入器，用于写均值矩阵，PCA旋转矩阵和PCA之后的数据集到文件系统
            //StreamWriter streamWriter1 = new StreamWriter(@"..\..\MNIST\mnist_train_small_HOG_PCA.csv");
            //StreamWriter streamWriter2 = new StreamWriter(@"..\..\MNIST\mnist_test_HOG_PCA.csv");
            //StreamWriter streamWriter3 = new StreamWriter(@"..\..\MNIST\PCA_HOG_TransformMatrix.csv");
            //StreamWriter streamWriter4 = new StreamWriter(@"..\..\MNIST\PCA_HOG_MeanVector.csv");

            //数据结构转换以便PCA使用，data 为转换后数据
            double[][] data = new double[inputDataTrain.Count][];
            for (int j = 0; j < inputDataTrain.Count; j++)
            {
                double[] row = new double[inputDataTrain[j].Length - 1];
                for (int i = 0; i < inputDataTrain[j].Length - 1; i++)
                {
                    row[i] = inputDataTrain[j][i + 1];
                }
                data[j] = row;
            }

            //假设 data 为 20000*784，即20000个样本，每个样本784维
            //矩阵转至，从 data（20000*784）到 dataT（784*20000）
            double[][] dataT = MatrixTranspose(data);
            //求每列均值
            double[] meanValue = data.Mean(0);

            //将均值向量写入文件系统
            string text = "";
            for (int i = 0; i < meanValue.Length; i++)
            {

                text += "," + meanValue[i].ToString("0.00000000");

            }
            //streamWriter4.WriteLine(text);
            //streamWriter4.Close();

            //dataTsubMean 为 data 矩阵每列减去其列均值得到的新矩阵。
            double[][] dataTsubMean = new double[data.Length][];
            for (int i = 0; i < data.Length; i++)
            {
                double[] row = new double[data[0].Length];
                for (int j = 0; j < data[0].Length; j++)
                {
                    row[j] = data[i][j] - meanValue[j];
                }
                dataTsubMean[i] = row;
            }

            //求解PCA旋转矩阵，取其前100个向量，result 为 784*100 的矩阵
            double[][] result = PCA(dataT, dataT, PCAAimDimRule.ItemsCount, 100f);

            //将PCA旋转矩阵写入文件系统
            for (int i = 0; i < result.Length; i++)
            {
                string lineText = "";
                for (int j = 0; j < result[i].Length; j++)
                {
                    lineText += "," + result[i][j].ToString("0.00000000");
                }
                //streamWriter3.WriteLine(lineText);
            }
            //streamWriter3.Close();

            //将原数据进行PCA重构，即 dataTsubMean*result， trainDataRePCA 为 20000* 100 的矩阵
            double[][] trainDataRePCA = MatrixMul(dataTsubMean, result);

            //将PCA之后的原数据写入文件系统
            for (int i = 0; i < trainDataRePCA.Length; i++)
            {
                string lineText = inputDataTrain[i][0].ToString();
                for (int j = 0; j < trainDataRePCA[i].Length; j++)
                {
                    lineText += "," + trainDataRePCA[i][j].ToString("0.00000000");
                }
                //streamWriter1.WriteLine(lineText);
            }
            //streamWriter1.Close();

            //以下是对于测试集的重构
            data = new double[inputDataTest.Count][];
            for (int j = 0; j < inputDataTest.Count; j++)
            {
                double[] row = new double[inputDataTest[j].Length - 1];
                for (int i = 0; i < inputDataTest[j].Length - 1; i++)
                {
                    row[i] = inputDataTest[j][i + 1];
                }
                data[j] = row;
            }

            dataTsubMean = new double[data.Length][];
            for (int i = 0; i < data.Length; i++)
            {
                double[] row = new double[data[0].Length];
                for (int j = 0; j < data[0].Length; j++)
                {
                    row[j] = data[i][j] - meanValue[j];
                }
                dataTsubMean[i] = row;
            }

            //将原来的旋转矩阵运用到测试集
            double[][] testDataRePCA = MatrixMul(dataTsubMean, result);

            //将PCA之后的测试集写入文件系统
            for (int i = 0; i < testDataRePCA.Length; i++)
            {
                string lineText = inputDataTest[i][0].ToString();
                for (int j = 0; j < testDataRePCA[i].Length; j++)
                {
                    lineText += "," + testDataRePCA[i][j].ToString("0.00000000");
                }
                //streamWriter2.WriteLine(lineText);
            }
            //streamWriter2.Close();
        }

        /// <summary>
        /// 训练SVM
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void BT_SVMTrain(object sender, EventArgs e)
        {
            //读取训练集
            List<double[]> inputDataTrain = ReadFileToList(@"..\..\MNIST\mnist_train_small_HOG_PCA.csv", ',');
            //样本维度为每行长度-1，因为每行第一的元素是标签，所以要去掉
            int dataDim = inputDataTrain[0].Length - 1;
            //获取训练集样本数量
            int trainSampleCount = inputDataTrain.Count;

            //OpenCV数据交换
            Emgu.CV.Matrix<float> trainData = new Emgu.CV.Matrix<float>(trainSampleCount, dataDim); //构造训练样本矩阵
            Emgu.CV.Matrix<int> trainClasses = new Emgu.CV.Matrix<int>(trainSampleCount, 1); //构造训练样本标签矩阵
            for (int i = 0; i < trainSampleCount; i++) //填充数据
            {
                for (int j = 0; j < dataDim; j++)
                {
                    trainData.Data[i, j] = (float)inputDataTrain[i][j + 1];
                }
                trainClasses[i, 0] = (int)inputDataTrain[i][0];
            }

            //构造SVM，并训练
            using (SVM svmClassifier = new SVM())
            {

                svmClassifier.SetKernel(SVM.SvmKernelType.Poly); //使用多项式核
                svmClassifier.Type = SVM.SvmType.CSvc; //使用多分类
                svmClassifier.C = 10; //多项式C参数
                svmClassifier.Gamma = 0.1; //多项式Gamma参数
                svmClassifier.Degree = 3; //D参数

                //训练
                TrainData td = new TrainData(trainData, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, trainClasses); 
                bool trained = svmClassifier.TrainAuto(td, 10);

                //保存训练结果到文件系统
                //svmClassifier.Save(@"..\..\MNIST\SVMTrainResult_HOG_PCA.xml");           
            }               
        }

        /// <summary>
        /// 用KNN测试测试集
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void BT_KNNTest(object sender, EventArgs e)
        {
            textBox1.Text = "读取数据...\r\n";
            List<double[]> inputDataTrain = ReadFileToList(@"..\..\MNIST\mnist_train_small_HOG.csv", ',');
            List<double[]> inputDataTest = ReadFileToList(@"..\..\MNIST\mnist_test_HOG.csv", ',');
            List<double[]> testDataRaw = ReadFileToList(@"..\..\MNIST\mnist_test.csv", ',');
            textBox1.AppendText("读取数据成功！\r\n");

            //进行KNN，结果为矩阵，每一行对应测试集中每个样本的测试结果，依次为距离其最近的K个训练集样本的标签
            List<double[]> result = KNN(inputDataTrain, inputDataTest, 10);

            textBox1.AppendText("正在统计结果...\r\n");
            //统计KNN结果，输出为三维矩阵，存放原始结果，和K个训练集样本的标签以及他们出现的次数和占比，按次数倒序排列
            List<List<double[]>> ratioMap = CalRatio(result);

            //正确率
            int accCount = 0;
            //索引
            int index = 0;
            textBox1.AppendText("正在筛选失败项目原图...\r\n");

            //判断是否分类正确
            foreach (var item in ratioMap)
            {
                if (item[0][0] == item[1][0])
                {
                    accCount++; //正确
                }
                else
                {
                    //将错误的对应图像存入文件系统
                    Bitmap failureBitmap = new Bitmap(28, 28);
                    for (int i = 1; i < testDataRaw[index].Length; i++)
                    {
                        failureBitmap.SetPixel((i - 1) % 28, (i - 1) / 28,
                            Color.FromArgb(255, (int)testDataRaw[index][i], (int)testDataRaw[index][i], (int)testDataRaw[index][i]));
                    }
                    failureBitmap.Save(@"..\..\MNIST\FailureSamples\KNN\" + testDataRaw[index][0] + "_" + item[1][0] + "_" + item[1][2] + "_" + index + ".bmp");
                }
                index++;
            }
            double acc = (double)accCount / inputDataTest.Count;
            textBox1.AppendText("成功：" + accCount + "个，失败" + (inputDataTest.Count - accCount) + "个\r\n正确率：" + acc * 100 + "%");
        }
        /// <summary>
        /// 用训练好的SVM测试测试集的正确率
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void BT_SVMTest(object sender, EventArgs e)
        {
            using (SVM svmClassifier = new SVM())
            {
                textBox1.Text = "读取数据...\r\n";
                List<double[]> testDataRaw = ReadFileToList(@"..\..\MNIST\mnist_test.csv", ',');
                List<double[]> inputDataTest = ReadFileToList(@"..\..\MNIST\mnist_test_HOG_PCA.csv", ',');
                textBox1.AppendText("读取数据成功！\r\n");

                textBox1.AppendText("读取支持向量...\r\n");
                svmClassifier.Load(@"..\..\MNIST\SVMTrainResult_HOG_PCA.xml");
                textBox1.AppendText("读取支持向量成功\r\n");

                int dataDim = inputDataTest[0].Length - 1;
                int testSampleCount = inputDataTest.Count;

                //OpenCV数据交换
                Emgu.CV.Matrix<float> testData = new Emgu.CV.Matrix<float>(testSampleCount, dataDim);
                Emgu.CV.Matrix<int> testClasses = new Emgu.CV.Matrix<int>(testSampleCount, 1);
                for (int i = 0; i < testSampleCount; i++)
                {
                    for (int j = 0; j < dataDim; j++)
                    {
                        testData.Data[i, j] = (float)inputDataTest[i][j + 1];
                    }
                    testClasses[i, 0] = (int)inputDataTest[i][0];
                }

                //正确率
                double acc = 0;

                //正确个数计数
                int Count = 0;
                string text = textBox1.Text;
                for (int i = 0; i < testSampleCount; i++)
                {
                    if (i % 100 == 0)
                    {
                        textBox1.Text = "";
                        textBox1.AppendText(text + "正在处理测试集：" + i.ToString() + "/" + testSampleCount + "\r\n");
                    }

                    //OpenCV数据交换，将测试数据转化为OpenCV使用的格式
                    Emgu.CV.Matrix<float> sample = new Emgu.CV.Matrix<float>(1, dataDim);
                    for (int j = 0; j < dataDim; j++)
                    {
                        sample.Data[0, j] = testData.Data[i, j];
                    }

                    //判断是否分类正确
                    if ((int)svmClassifier.Predict(sample) == testClasses[i, 0])
                    {
                        Count++; //正确
                    }
                    else
                    {
                        //不正确，筛选出此时对应的图像，转换为Bitmap后存入文件系统
                        Bitmap failureBitmap = new Bitmap(28, 28);
                        for (int k = 1; k < testDataRaw[i].Length; k++)
                        {
                            failureBitmap.SetPixel((k - 1) % 28, (k - 1) / 28,
                                Color.FromArgb(255, (int)testDataRaw[i][k], (int)testDataRaw[i][k], (int)testDataRaw[i][k]));
                        }
                        failureBitmap.Save(@"..\..\MNIST\FailureSamples\SVM\" + testDataRaw[i][0] + "_" + svmClassifier.Predict(sample) + "_" + i.ToString() + ".bmp");
                    }
                    acc = (double)Count / (double)testSampleCount;
                }
                textBox1.AppendText("正在筛选失败项目原图...\r\n");
                textBox1.AppendText("成功：" + Count + "个，失败" + (testSampleCount - Count) + "个\r\n正确率：" + acc * 100 + "%");
            }
        }

        /// <summary>
        /// KNN和SVM联合测试
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void BT_United(object sender, EventArgs e)
        {
            using (SVM svmClassifier = new SVM())
            {


                double acc = 0;
                textBox1.Text = "读取数据...\r\n";
                List<double[]> inputDataTrain = ReadFileToList(@"..\..\MNIST\mnist_train_small_HOG.csv", ',');
                List<double[]> inputDataTestKNN = ReadFileToList(@"..\..\MNIST\mnist_test_HOG.csv", ',');
                List<double[]> inputDataTestSVM = ReadFileToList(@"..\..\MNIST\mnist_test_HOG_PCA.csv", ',');
                textBox1.AppendText("读取数据成功！\r\n");

                textBox1.AppendText("读取支持向量...\r\n");
                svmClassifier.Load(@"..\..\MNIST\SVMTrainResult_HOG_PCA.xml");
                textBox1.AppendText("读取支持向量成功\r\n");

                int dataDim = inputDataTestSVM[0].Length - 1;
                int testSampleCount = inputDataTestSVM.Count;

                Emgu.CV.Matrix<float> testData = new Emgu.CV.Matrix<float>(testSampleCount, dataDim);
                Emgu.CV.Matrix<int> testClasses = new Emgu.CV.Matrix<int>(testSampleCount, 1);

                for (int i = 0; i < testSampleCount; i++)
                {
                    for (int j = 0; j < dataDim; j++)
                    {
                        testData.Data[i, j] = (float)inputDataTestSVM[i][j + 1];
                    }
                    testClasses[i, 0] = (int)inputDataTestSVM[i][0];
                }

                int Count = 0;

                List<double[]> result = KNN(inputDataTrain, inputDataTestKNN, 10);
                textBox1.AppendText("正在整理数据...\r\n");
                List<List<double[]>> ratioMap = CalRatio(result);

                string text = textBox1.Text;
                for (int i = 0; i < testSampleCount; i++)
                {
                    if (i % 100 == 0)
                    {
                        textBox1.Text = "";
                        textBox1.AppendText(text + "正在处理测试集：" + i.ToString() + "/" + testSampleCount + "\r\n");
                    }

                    //整合结果的时候有区别
                    double resultFinal = -1;
                    Emgu.CV.Matrix<float> sample = new Emgu.CV.Matrix<float>(1, dataDim);
                    for (int j = 0; j < dataDim; j++)
                    {
                        sample.Data[0, j] = testData.Data[i, j];
                    }
                    int resultSVM = (int)svmClassifier.Predict(sample);

                    if ((int)resultSVM != (int)ratioMap[i][1][0])
                    {
                        //由于KNN对于1和6的检测概率比较高，所以在某些情况下可以选择更倾向KNN
                        if (ratioMap[i][1][0] == 1 && (resultSVM == 2 || resultSVM == 3 || resultSVM == 4 || resultSVM == 5 || resultSVM == 8))
                        {
                            resultFinal = ratioMap[i][1][0];
                        }
                        else
                        {
                            resultFinal = resultSVM;
                        }
                    }
                    else
                    {
                        resultFinal = resultSVM;
                    }
                    if (resultFinal == testClasses[i, 0])
                    {
                        Count++;
                    }
                }
                acc = (double)Count / (double)testSampleCount;
                textBox1.AppendText("正在对比数据...\r\n");
                textBox1.AppendText("成功：" + Count + "个，失败" + (testSampleCount - Count) + "个\r\n正确率：" + acc * 100 + "%");
            }
        }
        /// <summary>
        /// 识别绘图区的图像
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void BT_Recognize(object sender, EventArgs e)
        {

            textBox1.Text = "正在解析图像...\r\n";
            //将绘图区图像转换成灰度图并反色
            Bitmap revGrayBit;
            if (!isImage)
            {
                revGrayBit = ColorReverse(ConvertToGray((Bitmap)pictureBox1.Image));
            }
            else
            {
                revGrayBit = Bitmap2Bin(ColorReverse(Bitmap2Bin(ConvertToGray((Bitmap)pictureBox1.Image), 127)), 127);
            }
            //获取图像ROI
            double[,] roi = GetROI(revGrayBit);
            if (roi[0, 0] == 0 && roi[0, 1] == 0 && roi[1, 0] == 0 && roi[1, 1] == 0)
            {
                MessageBox.Show("先画图!");
                return;
            }

            //画出ROI区域
            Graphics graphics = Graphics.FromImage(pictureBox1.Image);
            Pen pen = new Pen(Color.BlueViolet, 5);
            graphics.DrawRectangle(pen, (float)roi[0, 0] - 10, (float)roi[0, 1] - 10, (float)roi[1, 0] - (float)roi[0, 0] + 20, (float)roi[1, 1] - (float)roi[0, 1] + 20);
            pictureBox1.Refresh();

            //将原图根据ROI裁剪并居中缩放到28*28的标准MNIST图像中
            double length = (Math.Max((int)roi[1, 0] - (int)roi[0, 0] + 1, (int)roi[1, 1] - (int)roi[0, 1] + 1)) / 5 * 6.3;
            int bias = ((int)length - Math.Min((int)roi[1, 0] - (int)roi[0, 0] + 1, (int)roi[1, 1] - (int)roi[0, 1] + 1)) / 2;
            int bias2 = (Math.Max((int)roi[1, 0] - (int)roi[0, 0] + 1, (int)roi[1, 1] - (int)roi[0, 1] + 1) - Math.Min((int)roi[1, 0] - (int)roi[0, 0] + 1, (int)roi[1, 1] - (int)roi[0, 1] + 1)) / 2;
            Bitmap tempBit = new Bitmap((int)length, (int)length);
            for (int i = 0; i < tempBit.Width; i++)
            {
                for (int j = 0; j < tempBit.Height; j++)
                {
                    tempBit.SetPixel(i, j, Color.FromArgb(255, 0, 0, 0));
                }
            }
            for (int i = 0; i < (int)roi[1, 0] - (int)roi[0, 0] + 1; i++)
            {
                for (int j = 0; j < (int)roi[1, 1] - (int)roi[0, 1] + 1; j++)
                {
                    if ((int)roi[1, 0] - (int)roi[0, 0] + 1 - (int)roi[1, 1] + (int)roi[0, 1] - 1 < 0)
                    {
                        tempBit.SetPixel(i + bias, j - bias2 + bias, revGrayBit.GetPixel(i + (int)roi[0, 0], j + (int)roi[0, 1]));
                    }
                    else
                    {
                        tempBit.SetPixel(i - bias2 + bias, j + bias, revGrayBit.GetPixel(i + (int)roi[0, 0], j + (int)roi[0, 1]));
                    }

                }
            }
            //显示处理后的图像
            pictureBox2.Image = ImageScale(tempBit, 28, 28);
            textBox1.AppendText("解析图像成功！\r\n");

            Bitmap testBit = (Bitmap)pictureBox2.Image;
            textBox1.AppendText("ROI: X0-" + roi[0, 0].ToString() + " Y0-" + roi[0, 1].ToString() + " X1-" + roi[1, 0].ToString() + " Y1-" + roi[1, 1].ToString() + "\r\n");

            textBox1.AppendText("正在进行HOG特征提取...\r\n");
            //进行HOG特征提取
            HOGDescriptor hog = new HOGDescriptor(new Size(28, 28),
                new Size(14, 14), new Size(7, 7), new Size(7, 7), 9);

            float[] hogVectorf = hog.Compute(new Image<Gray, byte>(testBit));
            double[] hogvector = new double[hogVectorf.Length];
            for (int i = 0; i < hogvector.Length; i++)
            {
                hogvector[i] = hogVectorf[i];
            }
            //组成KNN所需的矩阵
            List<double[]> knnMat = new List<double[]>();
            knnMat.Add(hogvector);

            textBox1.AppendText("HOG特征提取成功！\r\n");

            textBox1.AppendText("正在读取PCA旋转矩阵与均值向量...\r\n");
            //从文件系统中读取PCA的参数
            double[][] meanValMat = ReadFileToArray2D(@"..\..\MNIST\PCA_HOG_MeanVector.csv", ',');
            double[][] PCATMatrix = ReadFileToArray2D(@"..\..\MNIST\PCA_HOG_TransformMatrix.csv", ',');
            textBox1.AppendText("PCA旋转矩阵与均值向量读取成功！\r\n");

            double[] meanValue = meanValMat[0];

            textBox1.AppendText("正在进行PCA重构...\r\n");
            //用以上读取的参数对HOG向量进行PCA重构
            double[][] dataTsubMean = new double[knnMat.Count][];
            for (int i = 0; i < knnMat.Count; i++)
            {
                double[] row = new double[knnMat[0].Length];
                for (int j = 0; j < knnMat[0].Length; j++)
                {
                    row[j] = knnMat[i][j] - meanValue[j];
                }
                dataTsubMean[i] = row;
            }
            //得到 testDataRePCA 为 1*100
            double[][] testDataRePCA = MatrixMul(dataTsubMean, PCATMatrix);
            textBox1.AppendText("PCA重构成功！\r\n");
            textBox1.AppendText("\r\n");

            textBox1.AppendText("正在进行KNN检测...\r\n");
            List<double[]> inputDataTrain = ReadFileToList(@"..\..\MNIST\mnist_train_small_HOG.csv", ',');
            //进行KNN，并统计
            List<double[]> knnResult = KNN(inputDataTrain, knnMat, 10);
            List<List<double[]>> knnRatioMap = CalRatio(knnResult);
            double resultKNN = knnRatioMap[0][1][0];
            //显示所有KNN结果
            for (int i = 1; i < knnRatioMap[0].Count; i++)
            {
                textBox1.Invoke(new Action(() =>
                {
                    textBox1.AppendText("KNN 结果_" + i.ToString() + ": " + knnRatioMap[0][i][0] + "    权重: " + knnRatioMap[0][i][2] + "\r\n");
                }
                ));

            }
            textBox1.AppendText("\r\n");
            textBox1.AppendText("正在进行SVM检测...\r\n");
            Emgu.CV.Matrix<float> sample = new Emgu.CV.Matrix<float>(1, testDataRePCA[0].Length);
            for (int j = 0; j < testDataRePCA[0].Length; j++)
            {
                sample.Data[0, j] = (float)testDataRePCA[0][j];
            }
            //进行SVM检测
            double resultSVM = SVMTester.Predict(sample);

            textBox1.Invoke(new Action(() =>
            {
                //显示检测结果
                textBox1.AppendText("SVM 结果: " + resultSVM + "\r\n");
            }
            ));

            //进行结果整合
            int resultFinal = -1;
            if ((int)resultSVM != (int)resultKNN)
            {
                if (resultKNN == 1 && (resultSVM == 2 || resultSVM == 3 || resultSVM == 4 || resultSVM == 5 || resultSVM == 8))
                {
                    resultFinal = (int)resultKNN;
                }
                else
                {
                    resultFinal = (int)resultSVM;
                }
            }
            else
            {
                resultFinal = (int)resultSVM;
            }

            textBox1.Invoke(new Action(() =>
            {
                //显示最终结果
                textBox1.AppendText("\r\n");
                textBox1.AppendText("最终结果: " + resultFinal + "\r\n");
            }
            ));

            Brush brush = Brushes.Red;


            graphics.DrawString(resultFinal.ToString(), new Font("微软雅黑", 60F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(134))), brush, (float)roi[0, 0] - 10, (float)roi[0, 1] - 10, StringFormat.GenericDefault);
            pictureBox1.Refresh();
        }

        /// <summary>
        /// 识别图像文件
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button13_Click(object sender, EventArgs e)
        {
            OpenFileDialog fileDialog = new OpenFileDialog();
            fileDialog.ShowDialog();

            if (fileDialog.FileName != "" && fileDialog.FileName != null)
            {
                pictureBox1.Image = Image.FromFile(fileDialog.FileName);
                isImage = true;
            }
        }

        /// <summary>
        /// 将原数据按照标签分为不同的文件（测试用）
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void BT_Groupping(object sender, EventArgs e)
        {
            StreamReader streamReader = new StreamReader(@"..\..\MNIST\mnist_test.csv");
            StreamWriter[] streamWriters = new StreamWriter[10];
            for (int i = 0; i < streamWriters.Length; i++)
            {
                string path = @"..\..\MNIST\Grouped\Test\KNN\o" + i.ToString() + ".csv";
                streamWriters[i] = new StreamWriter(path);
            }

            while (!streamReader.EndOfStream)
            {
                string lineText = streamReader.ReadLine();
                streamWriters[Convert.ToInt32(lineText.Split(',')[0])].WriteLine(lineText);
            }

            streamReader.Close();
            foreach (var item in streamWriters)
            {
                item.Close();
            }
        }

        /// <summary>
        /// 将数据集二值化（测试用，实际未使用）
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void BT_Binaryzation(object sender, EventArgs e)
        {
            StreamReader streamReader1 = new StreamReader(@"..\..\MNIST\mnist_train_small.csv");
            StreamReader streamReader2 = new StreamReader(@"..\..\MNIST\mnist_test.csv");
            StreamWriter streamWriter1 = new StreamWriter(@"..\..\MNIST\mnist_train_small_binary.csv");
            StreamWriter streamWriter2 = new StreamWriter(@"..\..\MNIST\mnist_test_binary.csv");

            int threshold = 127; //阈值
            while (!streamReader1.EndOfStream)
            {
                string[] lineText = streamReader1.ReadLine().Split(',');
                string lineTextP = "";
                lineTextP += lineText[0] + ",";
                for (int i = 1; i < lineText.Length; i++)
                { 
                    if (i != lineText.Length - 1)
                    {
                        if (Convert.ToInt32(lineText[i]) <= threshold)
                        {
                            lineTextP += "0,";
                        }
                        else
                        {
                            lineTextP += "1,";
                        }
                    }
                    else
                    {
                        if (Convert.ToInt32(lineText[i]) <= threshold)
                        {
                            lineTextP += "0";
                        }
                        else
                        {
                            lineTextP += "1";
                        }
                    }
                }
                streamWriter1.WriteLine(lineTextP);
            }
            streamWriter1.Close();

            while (!streamReader2.EndOfStream)
            {
                string[] lineText = streamReader2.ReadLine().Split(',');
                string lineTextP = "";
                lineTextP += lineText[0] + ",";
                for (int i = 1; i < lineText.Length; i++)
                {  
                    if (i != lineText.Length - 1)
                    {
                        if (Convert.ToInt32(lineText[i]) <= threshold)
                        {
                            lineTextP += "0,";
                        }
                        else
                        {
                            lineTextP += "1,";
                        }
                    }
                    else
                    {
                        if (Convert.ToInt32(lineText[i]) <= threshold)
                        {
                            lineTextP += "0";
                        }
                        else
                        {
                            lineTextP += "1";
                        }
                    }
                }
                streamWriter2.WriteLine(lineTextP);
            }
            streamWriter2.Close();
        }

        private void button5_Click(object sender, EventArgs e)
        {
            double[][] a = new double[3][] { new double[] { 2, 1 }, new double[] { 2, 2 }, new double[] { 3, 1 } };
            double[][] b = MatrixTranspose(a);
            IEnumerable<string> fileNames = Directory.EnumerateFiles(@"C:\MediaSlot\CloudDocs\Docs\Course\Pattern Recognition\Course Project\Codes\CodesDotNet\PatternRecognitionCodes\PatternRecognitionCodes\MNIST\FailureSamples\SVM");
            Bitmap bitmap = new Bitmap(280, 280);
            for (int i = 0; i < fileNames.Count<string>(); i++)
            {
                Bitmap px = (Bitmap)Bitmap.FromFile(fileNames.ToList()[i]);
                for (int j = 0; j < 784; j++)
                {
                    bitmap.SetPixel(28 * (i % 10) + j % 28, 28 * (i / 10) + j / 28, px.GetPixel(j % 28, j / 28));
                }
            }
            bitmap.Save(@"C:\MediaSlot\CloudDocs\Docs\Course\Pattern Recognition\Course Project\Codes\CodesDotNet\PatternRecognitionCodes\PatternRecognitionCodes\MNIST\1.bmp");


        }

        /// <summary>
        /// 清理作图区
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button12_Click(object sender, EventArgs e)
        {

            pictureBox1.Image = new Bitmap(pictureBox1.Width, pictureBox1.Height);
            g = Graphics.FromImage(pictureBox1.Image);
            g.Clear(Color.White);
            textBox1.Clear();
            pictureBox1.Refresh();
            isImage = false;
        }

        /// <summary>
        /// 鼠标按下后绘图
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void pictureBox1_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
                lineStartX = e.X;
                lineStartY = e.Y;
                drawkine = true;
            }
        }

        /// <summary>
        /// 鼠标移动时显示图像
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void pictureBox1_MouseMove(object sender, MouseEventArgs e)
        {
            if (drawkine)
            {
                Pen p = new Pen(Color.Black, 25);
                g.DrawLine(p, lineStartX, lineStartY, e.X, e.Y);
                lineStartX = e.X;
                lineStartY = e.Y;
            }
            pictureBox1.Refresh();
            
        }

        /// <summary>
        /// 鼠标抬起时，停止绘图
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void pictureBox1_MouseUp(object sender, MouseEventArgs e)
        {
            drawkine = false;
        }

        #endregion

        #region Uni Methods

        /// <summary>
        /// 图像二值化
        /// </summary>
        /// <param name="bitmap"></param>
        /// <param name="threshold">阈值0~255</param>
        /// <returns></returns>
        private Bitmap Bitmap2Bin(Bitmap bitmap, int threshold)
        {
            Bitmap finalBitmap = new Bitmap(bitmap.Width, bitmap.Height);
            for (int i = 0; i < bitmap.Width; i++)
            {
                for (int j = 0; j < bitmap.Height; j++)
                {
                    Color color = bitmap.GetPixel(i, j).R < threshold ? Color.Black : bitmap.GetPixel(i, j);
                    finalBitmap.SetPixel(i, j, color);
                }
            }
            return finalBitmap;
        }

        /// <summary>
        /// 图像颜色反转
        /// </summary>
        /// <param name="bitmap">原图像</param>
        /// <returns></returns>
        private Bitmap ColorReverse(Bitmap bitmap)
        {
            Bitmap bitmapR = new Bitmap(bitmap.Width, bitmap.Height);
            for (int i = 0; i < bitmap.Width; i++)
            {
                for (int j = 0; j < bitmap.Height; j++)
                {
                    bitmapR.SetPixel(i, j, 
                        Color.FromArgb(bitmap.GetPixel(i, j).A, 
                        255 - bitmap.GetPixel(i, j).R, 255 - bitmap.GetPixel(i, j).G, 255 - bitmap.GetPixel(i, j).B));
                }
            }
            return bitmapR;
        }

        /// <summary>
        /// 图像缩放
        /// </summary>
        /// <param name="bitmap">原图像</param>
        /// <param name="width">目标图像的宽</param>
        /// <param name="height">目标图像的高</param>
        /// <returns></returns>
        private static Bitmap ImageScale(Bitmap bitmap, int width, int height)
        {
            Bitmap bmp = new Bitmap(width, height);
            Graphics g = Graphics.FromImage(bmp);
            g.Clear(Color.White);
            g.ScaleTransform((float)width/ (float)bitmap.Width, (float)height / (float)bitmap.Height);
            g.DrawImage(bitmap, 0, 0, bitmap.Width, bitmap.Height);
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
            return bmp;
        }

        /// <summary>
        /// ARGB转化为灰度图
        /// </summary>
        /// <param name="rgbImage">RGB图像</param>
        /// <returns></returns>
        private static Bitmap ConvertToGray(Bitmap rgbImage)
        {
            Bitmap grayImage = new Bitmap(rgbImage.Width, rgbImage.Height);
            for (int i = 0; i < rgbImage.Width; i++)
            {
                for (int j = 0; j < rgbImage.Height; j++)
                {
                    short gray = (short)((rgbImage.GetPixel(i, j).R * 19595 + rgbImage.GetPixel(i, j).G * 38469 + rgbImage.GetPixel(i, j).B * 7472) >> 16);
                    Color grayColor = Color.FromArgb(gray, gray, gray);
                    grayImage.SetPixel(i, j, grayColor);
                }
            }

            return grayImage;

        }

        /// <summary>
        /// 获取图像ROI
        /// </summary>
        /// <param name="bitmap"></param>
        /// <returns></returns>
        private double[,] GetROI(Bitmap bitmap)
        {
            double[,] ROI = new double[2, 2];
            bool inside = false;
            for (int i = 0; i < bitmap.Width; i++)
            {
                for (int j = 0; j < bitmap.Height; j++)
                {
                    if (!inside)
                    {
                        if (bitmap.GetPixel(i, j).R != 0)
                        {
                            ROI[0, 0] = i;
                            inside = true;
                        }
                    }
                    if (inside)
                    {
                        if (bitmap.GetPixel(i, j).R != 0)
                        {
                            ROI[1, 0] = i;
                            continue;
                        }
                    }
                }
            }

            inside = false;
            for (int i = 0; i < bitmap.Height; i++)
            {
                for (int j = 0; j < bitmap.Width; j++)
                {
                    if (!inside)
                    {
                        if (bitmap.GetPixel(j, i).R != 0)
                        {
                            ROI[0, 1] = i;
                            inside = true;
                        }
                    }
                    if (inside)
                    {
                        if (bitmap.GetPixel(j, i).R != 0)
                        {
                            ROI[1, 1] = i;
                            continue;
                        }
                    }
                }
            }

            return ROI;
        }

        /// <summary>
        /// 计算PCA旋转矩阵
        /// </summary>
        /// <param name="data">PCA训练数据</param>
        /// <param name="fitData">要使用PCA训练结果的数据</param>
        /// <param name="aimDimRule">决定要怎样输出特征向量</param>
        /// <param name="para">按权重输出特征向量的权重或者按数量输出特征向量的数量</param>
        /// <returns>旋转矩阵</returns>
        private double[][] PCA(double[][] data, double[][] fitData, PCAAimDimRule aimDimRule, float para)
        {
            double[][] output = null;
            //实例化PCA训练器
            PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis()
            {
                Method = PrincipalComponentMethod.Center, //均值PCA
                Whiten = true //数据规范化

            };

            //训练
            MultivariateLinearRegression transform = pca.Learn(data);

            //按照输出方式不同输出
            if (aimDimRule == PCAAimDimRule.All)
            {
                output = pca.Transform(fitData);
            }

            if (aimDimRule == PCAAimDimRule.ItemsCount)
            {
                pca.NumberOfOutputs = (int)para;
                output = pca.Transform(fitData);
            }

            if (aimDimRule == PCAAimDimRule.Ratio)
            {
                pca.ExplainedVariance = para;
                output = pca.Transform(fitData);
            }

            return output;
        }

        /// <summary>
        /// KNN算法
        /// </summary>
        /// <param name="trainSet">训练集</param>
        /// <param name="testSet">测试集</param>
        /// <param name="k">K的值</param>
        /// <returns>对应于测试集的结果矩阵</returns>
        private List<double[]> KNN(List<double[]> trainSet, List<double[]> testSet, int k)
        {

            List<double[]> resultSet = new List<double[]>();
            string text = textBox1.Text;

            //对于每个测试集中的样本，计算他和训练集中每个样本的距离
            for (int i = 0; i < testSet.Count; i++)
            {
                if (i % 10 == 0)
                {
                    textBox1.Text = "";
                    textBox1.AppendText(text + "正在处理测试集：" + i.ToString() + "/" + testSet.Count + "\r\n");
                }

                //建立层数为K的队列
                List<double[]> tempStack = new List<double[]>();

                //对于每个训练集样本
                for (int j = 0; j < trainSet.Count; j++)
                {
                    //计算两者距离
                    double distance = CalDistance(testSet[i], trainSet[j], 1, DistType.Norm_2);

                    if (j < k)
                    {
                        //最开始的k个距离把队列填满 {样本序号，距离}
                        tempStack.Add(new double[] { trainSet[j][0], distance });
                    }
                    else
                    {
                        if (j == k)
                        {
                            //填满后，对队列中的元素按照距离排序
                            BubbleSort(tempStack, SortingRule.ASC);
                        }
                        if (distance < tempStack[k - 1][1])
                        {
                            //对于新的距离，如果它小于队列中的最大距离，则将其加入队列，并删除最大距离对应的元素
                            tempStack[k - 1] = new double[] { trainSet[j][0], distance };
                            //重新排序
                            BubbleSort(tempStack, SortingRule.ASC);
                        }
                    }
                }

                //将数据序列化后输出
                double[] nearests = new double[tempStack.Count + 1];
                for (int ii = 0; ii < nearests.Length - 1; ii++)
                {
                    nearests[ii + 1] = tempStack[ii][0];
                }
                nearests[0] = testSet[i][0];
                resultSet.Add(nearests);
            }
            return resultSet;
        }

        /// <summary>
        /// 冒泡排序
        /// </summary>
        /// <param name="unSorted">需要排序的数据</param>
        /// <param name="sortingRule">排序规则</param>
        private void BubbleSort(List<double[]> unSorted, SortingRule sortingRule)
        {

            switch (sortingRule)
            {
                case SortingRule.ASC:
                    {
                        for (int i = 0; i < unSorted.Count; i++)
                        {
                            for (int j = i; j < unSorted.Count; j++)
                            {
                                if (unSorted[i][1] > unSorted[j][1])
                                {
                                    double[] temp = unSorted[i];
                                    unSorted[i] = unSorted[j];
                                    unSorted[j] = temp;
                                }
                            }
                        }
                    }
                    break;
                case SortingRule.DESC:
                    {
                        for (int i = 0; i < unSorted.Count; i++)
                        {
                            for (int j = i; j < unSorted.Count; j++)
                            {
                                if (unSorted[i][1] < unSorted[j][1])
                                {
                                    double[] temp = unSorted[i];
                                    unSorted[i] = unSorted[j];
                                    unSorted[j] = temp;
                                }
                            }
                        }
                    }
                    break;
                default:
                    break;
            }

        }

        /// <summary>
        /// 计算样本间的距离
        /// </summary>
        /// <param name="left">样本1</param>
        /// <param name="right">样本2</param>
        /// <param name="initSeq">起始序号</param>
        /// <param name="distType">距离类型</param>
        /// <returns></returns>
        private double CalDistance(double[] left, double[] right, int initSeq, DistType distType)
        {
            double distance = 0;
            switch (distType)
            {
                case DistType.Norm_1://街坊距离
                    {
                        for (int i = initSeq; i < left.Length; i++)
                        {
                            distance += Math.Abs(left[i] - right[i]);
                        }
                    }
                    break;
                case DistType.Norm_2://欧氏距离
                    {
                        for (int i = initSeq; i < left.Length; i++)
                        {
                            distance += (left[i] - right[i]) * (left[i] - right[i]);
                        }
                        distance = Math.Sqrt(distance);
                    }
                    break;
                case DistType.Hamming://汉明距离
                    {
                        for (int i = initSeq; i < left.Length; i++)
                        {
                            if (left[i] != right[i])
                            {
                                distance += 1;
                            }
                        }
                    }
                    break;
                case DistType.Cosine://余弦距离
                    {
                        double sum = 0, normL = 0, normR = 0;
                        for (int i = initSeq; i < left.Length; i++)
                        {
                            sum += left[i] * right[i];
                            normL += left[i] * left[i];
                            normR += right[i] * right[i];
                        }
                        distance = 1 - sum / Math.Sqrt(normL * normR);
                    }
                    break;
                default:
                    break;
            }
            return distance;
        }

        /// <summary>
        /// 统计KNN结果中每项占比
        /// </summary>
        /// <param name="resultSet">KNN的输出结果</param>
        /// <returns></returns>
        private List<List<double[]>> CalRatio(List<double[]> resultSet)
        {
            List<List<double[]>> ratioMap = new List<List<double[]>>();
            foreach (var item in resultSet)
            {
                DataTable dataTable = new DataTable
                {
                    TableName = item[0].ToString()
                };
                dataTable.Rows.Add(dataTable.NewRow());
                dataTable.Rows.Add(dataTable.NewRow());
                for (int i = 1; i < item.Length; i++)
                {
                    if (!dataTable.Columns.Contains(item[i].ToString()))
                    {
                        dataTable.Columns.Add(item[i].ToString());
                        dataTable.Rows[0][item[i].ToString()] =
                            (dataTable.Rows[0][item[i].ToString()] == DBNull.Value ?
                            0 : Convert.ToInt32(dataTable.Rows[0][item[i].ToString()])) + 1;
                    }
                    else
                    {
                        dataTable.Rows[0][item[i].ToString()] =
                            (dataTable.Rows[0][item[i].ToString()] == DBNull.Value ?
                            0 : Convert.ToInt32(dataTable.Rows[0][item[i].ToString()])) + 1;
                    }
                }
                for (int i = 0; i < dataTable.Columns.Count; i++)
                {
                    dataTable.Rows[1][i] = Convert.ToDouble(dataTable.Rows[0][i]) / (double)(item.Length - 1);
                }
                List<double[]> tempTable = new List<double[]>();
                for (int i = 0; i < dataTable.Columns.Count; i++)
                {
                    tempTable.Add(new double[] {
                           Convert.ToDouble(dataTable.Columns[i].ColumnName),
                           Convert.ToDouble(dataTable.Rows[0][i]),
                           Convert.ToDouble(dataTable.Rows[1][i])
                    });

                }
                BubbleSort(tempTable, SortingRule.DESC);
                tempTable.Insert(0, new double[] { Convert.ToDouble(dataTable.TableName), 0, 0 });

                ratioMap.Add(tempTable);
            }
            return ratioMap;
        }

        /// <summary>
        /// 矩阵乘法
        /// </summary>
        /// <param name="left">左矩阵</param>
        /// <param name="right">右矩阵</param>
        /// <returns></returns>
        private double[][] MatrixMul(double[][] left, double[][] right)
        {
            double[][] result = new double[left.Length][];
            if (left[0].Length == right.Length)
            {
                for (int i = 0; i < left.Length; i++)
                {
                    double[] row = new double[right[0].Length];
                    for (int j = 0; j < right[0].Length; j++)
                    {
                        for (int k = 0; k < right.Length; k++)
                        {
                            row[j] += left[i][k] * right[k][j];
                        }
                    }
                    result[i] = row;
                }
            }
            return result;
        }

        /// <summary>
        /// 矩阵转置
        /// </summary>
        /// <param name="ori">原矩阵</param>
        /// <returns></returns>
        private double[][] MatrixTranspose(double[][] ori)
        {
            double[][] result = new double[ori[0].Length][];
            for (int j = 0; j < ori[0].Length; j++)
            {
                double[] row = new double[ori.Length];

                for (int i = 0; i < ori.Length; i++)
                {
                    row[i] = ori[i][j];
                }

                result[j] = row;
            }



            return result;
        }

        /// <summary>
        /// 读取文本文件，存入二维数组中
        /// </summary>
        /// <param name="fileName">文件路径</param>
        /// <param name="splitChar">分隔符</param>
        /// <returns></returns>
        private double[][] ReadFileToArray2D(string fileName, char splitChar)
        {
            StreamReader streamReader = new StreamReader(fileName);
            List<double[]> tempList = new List<double[]>();

            while (!streamReader.EndOfStream)
            {
                string lineText = streamReader.ReadLine();
                double[] vector = new double[lineText.Split(splitChar).Length];
                for (int i = 0; i < lineText.Split(splitChar).Length; i++)
                {
                    vector[i] = Convert.ToDouble(lineText.Split(splitChar)[i]);
                }
                tempList.Add(vector);
            }

            streamReader.Close();

            double[][] data = new double[tempList.Count][];
            for (int i = 0; i < tempList.Count; i++)
            {
                data[i] = tempList[i];
            }
            return data;
        }

        /// <summary>
        /// 读取文本文件，存入数组列表中
        /// </summary>
        /// <param name="fileName">文件路径</param>
        /// <param name="splitChar">分隔符</param>
        /// <returns></returns>
        private List<double[]> ReadFileToList(string fileName, char splitChar)
        {
            StreamReader matFile = new StreamReader(fileName);

            List<double[]> mat = new List<double[]>();

            while (!matFile.EndOfStream)
            {
                string[] data = matFile.ReadLine().Split(splitChar);
                double[] lineData = new double[data.Length];
                for (int i = 0; i < data.Length; i++)
                {
                    lineData[i] = Convert.ToDouble(data[i]);
                }
                mat.Add(lineData);

            }

            matFile.Close();
            return mat;
        }

        #endregion  
    }



    /// <summary>
    /// PCA向量的输出方式
    /// </summary>
    enum PCAAimDimRule
    {
        /// <summary>
        /// 按比例输出
        /// </summary>
        Ratio,
        /// <summary>
        /// 按设定项目的数目输出
        /// </summary>
        ItemsCount,
        /// <summary>
        /// 全部输出
        /// </summary>
        All
    }

   /// <summary>
   /// 使用何种距离
   /// </summary>
    enum DistType
    {
        /// <summary>
        /// 一范数，街坊距离
        /// </summary>
        Norm_1,
        /// <summary>
        /// 二范数，欧氏距离
        /// </summary>
        Norm_2,
        /// <summary>
        /// 汉明距离
        /// </summary>
        Hamming,
        /// <summary>
        /// 余弦距离
        /// </summary>
        Cosine
    }
    /// <summary>
    /// 排序类型
    /// </summary>
    enum SortingRule
    {
        /// <summary>
        /// 顺序
        /// </summary>
        ASC,
        /// <summary>
        /// 倒序
        /// </summary>
        DESC
    }
}
