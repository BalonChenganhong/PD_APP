### version 0.4  

### date 2025.6.11  


# 新加功能：
## LoginActivity:

登录功能（username=user1；password=1）；登录成功后自动跳转到FoGDetectionActivity

## FoGDetectionActivity：

请求调用SMV进行冻结步态识别，返回冻结步态标签、最长冻结步态窗口数量;跳转到VideoDetectionActivity或者ChatActivity

## VideoDetectionActivity：

请求调用视频检测算法进行对指帕金森诊断，返回诊断结果

## ChatActivity:

请求发送POST给LLM的API，返回LLM的回传信息

