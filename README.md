# About-mobile-deep-learning
Record the process of a mobile deep learning project I did. The main content is to introduce how to realize mobile deep learning

想借之前做的一个MDL（mobile deep learning）的项目来讲讲MDL的操作过程：  
首先说明我的环境，因为设备问题，最新的pytorch使用不了，所以使用的是TensorFlow lite，相关环境：  
Python-3.6.0  
TensorFlow-2.2.0（TF2.2.0中已经带有keras了，可以在导入时使用`“import TensorFlow.keras as keras”`  来简洁使用，这与之前单独的keras库没有太大差别，这样就不需要另外安装keras库，）  
Android studio-3.2


# 一、	模型
模型可以使用TF或者keras框架进行训练和保存，与正常的模型训练没有区别。
训练完成后，由于TFlite使用的模型是.tflite模型，TF的模型保存后为.pb，keras模型保存后为.h5，所以需要将有关模型转换成.tflite模型。使用以下代码：  
```
model=load_model(model_path)  
output_graph_name = "{0}.tflite".format(lite_path)  
converter = tensorflow.lite.TFLiteConverter.from_keras_model(model=model)  
tflite_model = converter.convert()  
open(output_graph_name, "wb").write(tflite_model)  
```

*model_path*和*lite_path*分别为原模型路径和lite模型想要保存的路径，以上代码主要内容为：加载模型，在原模型基础上制作转换器，执行转换生成lite模型，保存文件。
代码执行成功，则会生成对应的以tflite为后缀的文件，这里可以注意观察，原模型和这个lite模型文件大小上的区别，lite模型比原模型小很多。

得到lite模型，这一部分完成。

# 二、	Android端准备工作
在Android studio中打开相关工程，首先在**app**目录下的*build.gradle*配置文件中进行编辑，  

1.在**dependencise**下加上以下包的引用，这是使Android项目支持Tflite的核心：  
`implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'`  

2.在android下加上以下代码，主要限制不要对Tflite模型进行压缩：  
```
//set no compress models
aaptOptions {
    noCompress "tflite"
}
```
**!!!编辑完成后，点击Sync now，重新build项目，确保成功再进行接下来的操作。!!!**  

在\app\src\main中创建一个名为“*assets*”的文件夹，系统会自动将其设为数据库文件夹，仔细棺材可以看到普通文件夹和数据库文件夹的图标有一些不同。
在此文件夹下，需要两个关键文件，一个是lite模型文件，即之前保存的后缀为tflite的文件，另一个是模型使用数据的所有label做成的txt文件，这个文件需要将模型中所有的label，一个一行地在txt文件写入，将两个文件放置“*assets*”的文件夹下。

这一部分完成。

# 三、	Android端代码编辑
这一部分的代码，我主要在主活动下编写，想要在其他地方调用，代码的逻辑是一样的，只是调用关系而已。
## 3.1 初始化变量和加载模型
```
private Interpreter tflite = null;

// load infer model
    private void load_model(String model) {
        try {
            tflite = new Interpreter(loadModelFile(model).asReadOnlyBuffer());
            Toast.makeText(MainActivity.this, model + " model load success", Toast.LENGTH_SHORT).show();
            Log.d(TAG, model + " model load success");
            load_result = true;
        } catch (IOException e) {
            Toast.makeText(MainActivity.this, model + " model load fail", Toast.LENGTH_SHORT).show();
            Log.d(TAG, model + " model load fail");
            load_result = false;
            e.printStackTrace();
        }
    }
```

```
/**
 * Memory-map the model file in Assets.
 */
private MappedByteBuffer loadModelFile(String model) throws IOException {
    AssetFileDescriptor fileDescriptor = getApplicationContext().getAssets().openFd(model + ".tflite");
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
}
```

以上代码执行后*tflite*变量中存放的为*assets*中的模型，可以用来后续进行预测。
## 3.2 初始化变量和加载label文件
```
private List<String> resultLabel = new ArrayList<>();
private void readCacheLabelFromLocalFile() {
    try {
        AssetManager assetManager = getApplicationContext().getAssets();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open("labels.txt")));
        String readLine = null;
        while ((readLine = reader.readLine()) != null) {
            resultLabel.add(readLine);
        }
        reader.close();
    } catch (Exception e) {
        Log.e("labelCache", "error " + e);
    }
}
```
以上代码执行后*resultLabel*变量中存放的为*assets*中的label文件，可以用来后续使用。
## 3.3 模型预测
主要代码：
```
private void PredictWav(String path){
        File f = new File(path);
        while (true){
            if(f.exists()){
                WaveData reader = new WaveData(path);
                double[][] tempdata = reader.getData();
                ByteBuffer inputData =getScaledMatrix(tempdata);
                try {
                    float[][] labelProbArray = new float[1][10];
                    long start = System.currentTimeMillis();
                    // get predict result
                    tflite.run(inputData, labelProbArray);
                    long end = System.currentTimeMillis();
                    long time = end - start;
                    float[] results = new float[labelProbArray[0].length];
                    System.arraycopy(labelProbArray[0], 0, results, 0, labelProbArray[0].length);
                    // show predict result and time
                    int[] r = get_max_result(results);
                    String show_text = " label：\n" + resultLabel.get(r[0]) +"\t\t\t\t\t\t\tProbability:\t\t"+ results[r[0]]*100+"%\n\nPredict Used:"+time + "ms";
                    result_text.setText(show_text);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                break;
            }
        }
    }
```
以下是相关函数：  

*getScaledMatrix*此函数为主要函数之一，负责将需要预测的数据文件进行数据提取，处理成模型能接受的数据类型，以下函数处理的是113\*113的txt文件，因为模型接收的是113\*113的矩阵数据，所以以下处理参数设置为113。首先将txt文件全部读取出，当做实参*fft_data*（因为我的矩阵数据就是FFT的数据所以命名这个，使用时根据数据定义变量名）传给函数，根据数据长度定义一个byte缓存区，长度为数据长度\*单个数据的byte位数，因为我使用的为float数据，所以是113\*113\*4的长度，提供一下表格参考（Java语言内）。
数据类型|byte|short|int|long|char|float|double
--|:--:|:--:|:--:|:--:|:--:|:--:|:--:
byte长度|1|2|4|8|2|4|8  

之后将*fft_data*数据按照从左到右，从上到下的顺序使用put函数写入到缓存区中，注意每种数据类型都有各自的put函数，可以先输入put根据代码补全进行查看。
```
public static ByteBuffer getScaledMatrix(double[][] fft_data) {
    ByteBuffer imgData = ByteBuffer.allocateDirect(113*113*4);
    imgData.order(ByteOrder.nativeOrder());
    for (int i = 0; i < 113; ++i) {
        for (int j = 0; j < 113; ++j) {
            final double val = fft_data[i][j];
            imgData.putFloat((float) val);
        }
    }
    return imgData;
}
```
对于*get_max_result*函数，首先要知道模型预测的结果是什么样的数据？首先模型在预测时，接收的标签数据不是单独的“label”标签，而是将label进行编码处理成**one-hot**码，再进行训练，例如有一个数据的label为“0 0 1 0 0”，则表示这条数据是第三类类别的概率为100%（因为这是已知数据，所以概率是100%），同样模型在预测时，预测的结果和这种数据差不多，例如一个预测结果为“0.01 0.01 0.95 0.01 0.02”，则表示这条数据被预测为五个类别的概率分别为1%，1%，95%，1%，2%，再从中取最大值，则为模型预测的最终类别。在*PredictWav*函数中的变量*labelProbArray*就是这样的数据，所以需要手动从中获取概率最大的类别序号，在根据序号去label.txt文件中获取类别名。
```
private int[] get_max_result(float[] result) {
    float probability = result[0];
    float sepro = result[0];
    int[] r = {0,0};
    for (int i = 0; i < result.length; i++) {
        if (probability < result[i]) {
            r[1] = r[0];
            sepro = probability;
            probability = result[i];
            r[0] = i;
        }
        else if(sepro<result[i]){
            r[1] = i;
            sepro = result[i];
        }
    }
    return r;
}
```

在*PredictWav*函数中使用`tflite.run(inputData, labelProbArray)`，直接将数据和空的概率数组传给模型变量，模型会对数据进行预测，将预测结果直接写入*labelProbArray*中。再使用`resultLabel.get(r[0])`可以获取label.txt中对应的预测类别名。
上诉代码因为我使用的是矩阵数据进行训练和预测，这里再补充一个使用图像的代码段。主要就是*getScaledMatrix*函数有所不同，其他地方的代码不需要进行改动。
```
public static ByteBuffer getScaledMatrix(Bitmap bitmap, int[] ddims) {
        ByteBuffer imgData = ByteBuffer.allocateDirect(ddims[0] * ddims[1] * ddims[2] * ddims[3] * 4);
        imgData.order(ByteOrder.nativeOrder());
        // get image pixel
        int[] pixels = new int[ddims[2] * ddims[3]];
        Bitmap bm = Bitmap.createScaledBitmap(bitmap, ddims[2], ddims[3], false);
        bm.getPixels(pixels, 0, bm.getWidth(), 0, 0, ddims[2], ddims[3]);
        int pixel = 0;
        for (int i = 0; i < ddims[2]; ++i) {
            for (int j = 0; j < ddims[3]; ++j) {
                final int val = pixels[pixel++];
                imgData.putFloat(((((val >> 16) & 0xFF) - 128f) / 128f));
                imgData.putFloat(((((val >> 8) & 0xFF) - 128f) / 128f));
                imgData.putFloat((((val & 0xFF) - 128f) / 128f));
            }
        }

        if (bm.isRecycled()) {
            bm.recycle();
        }
        return imgData;
}
```
函数的参数*bitmap*使用`PhotoUtil.getScaleBitmap(image_path)`进行获取，*image_path*为图片的路径，*ddims*为模型训练和测试的图片尺寸数据，进行统一定义，比如：  
`private int[] ddims = {1, 3, 28, 28}  `
表示1张图片，图片为3通道，图片大小为28\*28，所以需要的缓冲区大小为*ddims[0] * ddims[1] * ddims[2] * ddims[3] * 4*，之后获取*bitmap*中的*pixels*数据，注意每一个*pixels*数据包括了**3**个通道的数值，所以需要对数据进行三次移位操作或者各个通道上的数值。最终将数据返回，供后续模型使用。

获取到数据的预测类别后还可以进行其他操作。

基于TFlite的Mobile Deep Learning主要操作介绍完毕。
