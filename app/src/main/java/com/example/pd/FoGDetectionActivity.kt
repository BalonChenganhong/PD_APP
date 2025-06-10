package com.example.pd

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import com.google.gson.Gson
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import java.io.IOException

class FoGDetectionActivity : AppCompatActivity() {

    private lateinit var btnRunPatient1: Button
    private lateinit var btnRunPatient2: Button
    private lateinit var tvResultFog: TextView
    private val client = OkHttpClient()
    private val BASE_URL = "http://10.12.51.208:5000" // 你的电脑IP地址

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_fog_detection)

        // 初始化UI组件
        btnRunPatient1 = findViewById(R.id.btn_run_smv_1)
        btnRunPatient2 = findViewById(R.id.btn_run_smv_2)
        tvResultFog = findViewById(R.id.tv_result_fog)
        // 设置登录按钮点击事件
        btnRunPatient1.setOnClickListener {
            val patient_id = "1169".trim()

            // 执行登录请求
            fog(patient_id)
        }
    }

    private fun fog(patient_id: String) {
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
            .url(BASE_URL + "/fog")
            .post(requestBody)
            .build()

        // 执行异步请求
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    tvResultFog.text = "请求失败: ${e.message}"
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
                    val total = stats["total"] as? Double ?: 0.0
                    val positiveCount = stats["positive_count"] as? Double ?: 0.0

                    // 格式化结果
                    val resultText = "FoG 检测结果：\n" +
                            "二进制预测：$binaryString\n" +
                            "总预测数：${total.toInt()}\n" +
                            "阳性预测数：${positiveCount.toInt()}\n"

                    runOnUiThread {
                        tvResultFog.text = resultText
                        Toast.makeText(this@FoGDetectionActivity, "检测完成", Toast.LENGTH_SHORT).show()
                    }
                } catch (e: Exception) {
                    runOnUiThread {
                        tvResultFog.text = "响应解析错误：${e.message}"
                    }
                }
            }
        })
    }
}

