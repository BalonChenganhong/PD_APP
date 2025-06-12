package com.example.pd

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import com.google.gson.Gson
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import java.io.IOException

class VideoDetectionActivity : AppCompatActivity() {

    private lateinit var btn_run_video_1: Button
    private lateinit var tvResultVideo: TextView
    private val client = OkHttpClient()
    private val BASE_URL = "http://10.12.51.208:5000" // 你的电脑IP地址

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_detection)

        // 初始化UI组件
        btn_run_video_1 = findViewById(R.id.btn_run_video_1)
        tvResultVideo = findViewById(R.id.tv_result_video)
        // 设置登录按钮点击事件
        btn_run_video_1.setOnClickListener {
            val patient_id = "1169".trim()

            // 执行登录请求
            video(patient_id)
        }
    }

    private fun video(patient_id: String) {
        // 创建请求体
        val requestBodyMap = mapOf(
            "patient_id" to patient_id,
        )

        val gson = Gson()
        val requestBodyJson = gson.toJson(requestBodyMap)

        // 修复: 使用扩展函数创建MediaType
        val requestBody = RequestBody.create(
            "application/json; charset=utf-8".toMediaTypeOrNull(),
            requestBodyJson
        )

        // 创建请求
        val request = Request.Builder()
            .url(BASE_URL + "/video")
            .post(requestBody)
            .build()

        // 执行异步请求
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    tvResultVideo.text = "请求失败: ${e.message}"
                }
            }

            override fun onResponse(call: Call, response: Response) {
                val responseData = response.body?.string() ?: ""
                // 解析返回的 JSON 数据，获取 message
                val gson = Gson()
                try {
                    val resultMap: Map<*, *>? = gson.fromJson(responseData, Map::class.java)

                    // 获取二进制字符串
                    val binaryString = resultMap?.get("message") as? String ?: ""

                    // 获取统计信息
                    val stats = resultMap?.get("stats") as? Map<String, Any> ?: emptyMap()
                    val probability = stats["probability"] as? Double ?: 0.0

                    // 格式化结果
                    val resultText = "FoG 检测结果：\n" +
                            "二进制预测：$binaryString\n" +
                            "预测概率：${probability.toFloat()}\n"

                    runOnUiThread {
                        tvResultVideo.text = resultText
                        Toast.makeText(this@VideoDetectionActivity, "检测完成", Toast.LENGTH_SHORT).show()
                    }
                } catch (e: Exception) {
                    runOnUiThread {
                        tvResultVideo.text = "响应解析错误：${e.message}"
                    }
                }
            }
        })
    }
}

