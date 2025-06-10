package com.example.pd

import android.content.Intent
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

class LoginActivity : AppCompatActivity() {

    private lateinit var etUsername: EditText
    private lateinit var etPassword: EditText
    private lateinit var btnLogin: Button
    private lateinit var tvResult: TextView
    private val client = OkHttpClient()
    private val BASE_URL = "http://10.12.51.208:5000" // 你的电脑IP地址

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login)

        // 初始化UI组件
        etUsername = findViewById(R.id.et_username)
        etPassword = findViewById(R.id.et_password)
        btnLogin = findViewById(R.id.btn_login)
        tvResult = findViewById(R.id.tv_result)

        // 设置登录按钮点击事件
        btnLogin.setOnClickListener {
            val username = etUsername.text.toString().trim()
            val password = etPassword.text.toString().trim()

            if (username.isEmpty() || password.isEmpty()) {
                Toast.makeText(this, "用户名和密码不能为空", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            // 执行登录请求
            login(username, password)
        }
    }

    private fun login(username: String, password: String) {
        // 创建请求体
        val requestBodyMap = mapOf(
            "username" to username,
            "password" to password
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
            .url(BASE_URL + "/login")
            .post(requestBody)
            .build()

        // 执行异步请求
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    tvResult.text = "请求失败: ${e.message}"
                }
            }

            override fun onResponse(call: Call, response: Response) {
                val responseData = response.body?.string() ?: ""
                // 解析返回的 JSON 数据，获取 message
                val gson = Gson()
                try {
                    val resultMap = gson.fromJson(responseData, Map::class.java)
                    val message = resultMap["message"] as? String ?: "未知响应"
                    runOnUiThread {
                        tvResult.text = message
                        // 也可以根据 message 做进一步交互，比如登录成功后跳转到其他页面
                        if (message == "登录成功") {
                            Toast.makeText(
                                this@LoginActivity,
                                "欢迎登录！",
                                Toast.LENGTH_SHORT
                            ).show()

                            // 跳转至 FoGDetectionActivity
                            val intent = Intent(this@LoginActivity, FoGDetectionActivity::class.java)

                            // 可选：传递登录用户信息（如用户名）到下个页面
                            intent.putExtra("USERNAME", username)

                            startActivity(intent)
                            finish() // 关闭当前登录页面，防止返回后重复请求
                        } else if (message == "用户名和密码不能为空") {
                            Toast.makeText(
                                this@LoginActivity,
                                "请填写完整的用户名和密码",
                                Toast.LENGTH_SHORT
                            ).show()
                        } else if (message == "用户名或密码错误") {
                            Toast.makeText(
                                this@LoginActivity,
                                "认证失败，请检查账号密码",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    }
                } catch (e: Exception) {
                    runOnUiThread {
                        tvResult.text = "响应解析错误：${e.message}"
                    }
                }
            }
        })
    }
}

