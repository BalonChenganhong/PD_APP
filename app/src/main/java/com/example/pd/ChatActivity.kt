package com.example.pd

import android.content.Intent
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

class ChatActivity : AppCompatActivity() {

    private lateinit var btnSend: Button
    private lateinit var tvResultChat: TextView
    private val client = OkHttpClient()
    private val BASE_URL = "http://10.12.51.208:5000" // 你的电脑IP地址

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_chat)

        // 初始化UI组件
        btnSend = findViewById(R.id.btn_send)
        tvResultChat = findViewById(R.id.tv_result_chat)

        // 设置登录按钮点击事件
        btnSend.setOnClickListener {
            val query = "check".trim()
            val response_mode = "blocking".trim()
            val conversation_id = "".trim()
            val user = "user1".trim()

            // 执行登录请求
            chat(query, response_mode, conversation_id, user)
        }

    }

    private fun chat(query: String,
                     response_mode: String,
                     conversation_id: String,
                     user: String) {
        // 创建请求体
        val requestBodyMap = mapOf(
            "query" to query,
            "response_mode" to response_mode,
            "conversation_id" to conversation_id,
            "user" to user
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
            .url(BASE_URL + "/api")
            .post(requestBody)
            .build()

        // 执行异步请求
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    tvResultChat.text = "请求失败: ${e.message}"
                }
            }

            override fun onResponse(call: Call, response: Response) {
                val responseData = response.body?.string() ?: ""
                // 解析返回的 JSON 数据，获取 message
                val gson = Gson()
                try {
                    val resultMap: Map<*, *>? = gson.fromJson(responseData, Map::class.java)

                    // 获取二进制字符串
                    val answer = resultMap?.get("answer") as? String ?: ""


                    // 格式化结果
                    val resultText = "LLM Answer：$answer\n"

                    runOnUiThread {
                        tvResultChat.text = resultText
                        Toast.makeText(this@ChatActivity, "获得LLM回复", Toast.LENGTH_SHORT).show()
                    }
                } catch (e: Exception) {
                    runOnUiThread {
                        tvResultChat.text = "响应解析错误：${e.message}"
                    }
                }
            }
        })
    }
}

