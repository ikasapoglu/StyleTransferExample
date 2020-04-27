package com.ikasapoglu.styletransfer_example

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.Image
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.HashMap

class MainActivity : AppCompatActivity() {

    lateinit var targetImage: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tv_select_target_img.setOnClickListener {
            val intent = Intent()
            intent.type = "image/*"
            intent.action = Intent.ACTION_GET_CONTENT
            startActivityForResult(intent, REQUEST_TARGET_IMG)
        }

        btn_apply.setOnClickListener {
            val transformOptions = Interpreter.Options()
                    .setNumThreads(24)

            val transformInterpreter = Interpreter(loadModelFile(this@MainActivity,
                    TRANSFORM_MODEL), transformOptions)

            val transferredImage = runTransform(transformInterpreter, targetImage)
            iv_result_img.setImageBitmap(transferredImage)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK) {
            val uri = data!!.data
            var bitmap = BitmapFactory.decodeStream(this.contentResolver.openInputStream(uri!!))
            bitmap = Bitmap.createScaledBitmap(bitmap, CONTENT_IMG_SIZE_WIDTH, CONTENT_IMG_SIZE_HEIGHT, false)
            iv_target_img!!.setImageBitmap(bitmap)
            targetImage = bitmap
        }
    }


    private fun runTransform(tflite: Interpreter, contentImage: Bitmap): Bitmap? {
        var inputTensorImage: TensorImage? = getInputTensorImage(tflite, contentImage)
        val inputs = arrayOfNulls<Any>(1)
        inputs[0] = inputTensorImage!!.buffer
        val outputShape = intArrayOf(DIM_BATCH_SIZE, CONTENT_IMG_SIZE_WIDTH, CONTENT_IMG_SIZE_HEIGHT, DIM_PIXEL_SIZE)
        val outputTensorBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
        val outputs: MutableMap<Int, Any> = HashMap()
        outputs[0] = outputTensorBuffer.buffer
        tflite.runForMultipleInputsOutputs(inputs, outputs)
        return convertOutputToBmp(outputTensorBuffer.floatArray)
    }

    private fun getInputTensorImage(tflite: Interpreter, inputBitmap: Bitmap): TensorImage {
        val imageDataType = tflite.getInputTensor(0).dataType()
        val inputTensorImage = TensorImage(imageDataType)

        inputTensorImage.load(inputBitmap)
        val imageProcessor: ImageProcessor = ImageProcessor.Builder().build()
        return imageProcessor.process(inputTensorImage)
    }

    private fun convertOutputToBmp(output: FloatArray): Bitmap {
        val result = Bitmap.createBitmap(
                CONTENT_IMG_SIZE_WIDTH, CONTENT_IMG_SIZE_HEIGHT, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(CONTENT_IMG_SIZE_WIDTH * CONTENT_IMG_SIZE_HEIGHT)
        val a = 0xFF shl 24
        var i = 0
        var j = 0
        while (j < output.size) {
            val r = output[j++].toInt()
            val g = output[j++].toInt()
            val b = output[j++].toInt()
            pixels[i] = a or (r shl 16) or (g shl 8) or b
            i++
        }

        result.setPixels(pixels, 0, CONTENT_IMG_SIZE_WIDTH, 0, 0, CONTENT_IMG_SIZE_WIDTH, CONTENT_IMG_SIZE_HEIGHT)
        return result
    }

    fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    companion object {
        private const val REQUEST_TARGET_IMG = 1
        private const val CONTENT_IMG_SIZE_WIDTH = 640
        private const val CONTENT_IMG_SIZE_HEIGHT = 480
        private const val DIM_BATCH_SIZE = 1
        private const val DIM_PIXEL_SIZE = 3

        val TRANSFORM_MODEL = "starry_night_640x480.tflite"
    }
}
