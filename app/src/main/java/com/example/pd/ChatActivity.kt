package com.example.pd

import android.content.Context
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

    private lateinit var btnRequestQ1: Button
    private lateinit var btnRequestQ2: Button
    private lateinit var btnRequestQ3: Button
    private lateinit var btnRequestQ5: Button
    private lateinit var btnRequestQ6: Button
    private lateinit var tvResultChat: TextView
    private lateinit var etDesperationQ1: EditText
    private lateinit var etDesperationQ2: EditText
    private lateinit var etDesperationQ3: EditText
    private lateinit var etDesperationQ5: EditText
    private lateinit var etDesperationQ6: EditText
    private val client = OkHttpClient()
    private val BASE_URL = "http://10.12.51.208:5000" // 你的电脑IP

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_chat)
        Log.d("ChatActivity", "onCreate called")

        // 初始化UI组件
        btnRequestQ1 = findViewById(R.id.btn_request_q1)
        btnRequestQ2 = findViewById(R.id.btn_request_q2)
        btnRequestQ3 = findViewById(R.id.btn_request_q3)
        btnRequestQ5 = findViewById(R.id.btn_request_q5)
        btnRequestQ6 = findViewById(R.id.btn_request_q6)
        tvResultChat = findViewById(R.id.tv_result_chat)
        etDesperationQ1 = findViewById(R.id.et_request_q1)
        etDesperationQ2 = findViewById(R.id.et_request_q2)
        etDesperationQ3 = findViewById(R.id.et_request_q3)
        etDesperationQ5 = findViewById(R.id.et_request_q5)
        etDesperationQ6 = findViewById(R.id.et_request_q6)
        Log.d("ChatActivity", "UI components initialized")

        // 设置登录按钮点击事件
        btnRequestQ1.setOnClickListener {
            val query = "这是冻结步态量表中的一个问题、打分的规则以及患者的主诉，请给出此患者的分数".trim()
            val description = etDesperationQ1.text.toString().trim()

            // 执行登录请求
            chat(query, description)
        }

        btnRequestQ2.setOnClickListener {
            val query = "这是冻结步态量表中的一个问题、打分的规则以及患者的主诉，请给出此患者的分数".trim()
            val description = etDesperationQ2.text.toString().trim()

            // 执行登录请求
            chat(query, description)
        }

        btnRequestQ3.setOnClickListener {
            val query = "这是冻结步态量表中的一个问题、打分的规则以及患者的主诉，请给出此患者的分数".trim()
            val description = etDesperationQ3.text.toString().trim()

            // 执行登录请求
            chat(query, description)
        }

        btnRequestQ5.setOnClickListener {
            val query = "这是冻结步态量表中的一个问题、打分的规则以及患者的主诉，请给出此患者的分数".trim()
            val description = etDesperationQ5.text.toString().trim()

            // 执行登录请求
            chat(query, description)
        }

        btnRequestQ6.setOnClickListener {
            val query = "这是冻结步态量表中的一个问题、打分的规则以及患者的主诉，请给出此患者的分数".trim()
            val description = etDesperationQ6.text.toString().trim()

            // 执行登录请求
            chat(query, description)
        }
        Log.d("ChatActivity", "Button click listeners set")

    }

    private fun chat(query: String,
                     description: String) {
        // 创建请求体
        val response_mode = "blocking".trim()
        val conversation_id = "".trim()
        val user = "user1".trim()
        val question_id = "1".trim()
        val requestBodyMap = mapOf(
            "query" to query,
            "response_mode" to response_mode,
            "conversation_id" to conversation_id,
            "user" to user,
            "description" to description,
            "question_id" to question_id
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
                    val answer = resultMap?.get("answer") as? String ?: "0"


                    // 格式化结果
                    val sharedPref = getSharedPreferences("fog_result", Context.MODE_PRIVATE)
                    with(sharedPref.edit()) {
                        putInt("score_q1", answer.toInt())
                        apply()
                    }
                    val resultText = "LLM Answer：${answer}\n"

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

