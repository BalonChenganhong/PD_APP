from flask import Flask, request, jsonify
import numpy as np
import torch
from torch.distributions.constraints import positive


app = Flask(__name__)

# 模拟用户数据库
users = {
    "user1": "1",
    "user2": "2"
}


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    print(f'get login request: user:{username};password:{password}')

    if not username or not password:
        return jsonify({"message": "用户名和密码不能为空"}), 400

    if username in users and users[username] == password:
        return jsonify({"message": "登录成功"}), 200
    else:
        return jsonify({"message": "用户名或密码错误"}), 401

@app.route('/fog', methods=['POST'])
def fog():
    message = request.get_json()
    patient_id = message.get('patient_id')

    print(f'get running fog network request: patient:{patient_id}')
    if patient_id == '1169':
        data = np.load("gait/1169_26_data.npy")
        label = np.load("gait/1169_26_label.npy")
    elif patient_id == '1180':
        data = np.load("gait/1180_26_data.npy")
        label = np.load("gait/1180_26_label.npy")

    model = torch.load("gait/net_CNN_1D_T.pth")
    model.eval()
    model.cuda()
    print("***Loading SMV***\n", model)
    data = torch.tensor(data, dtype=torch.float32).cuda()

    y_hat = model(data).cpu().detach().numpy()


    predictions = (y_hat > 0.5).astype(int).tolist()
    predictions = [item[0] for item in predictions]

    predictions_str = ''.join(map(str, predictions))
    total = len(predictions)
    longest_fog = find_longest_consecutive_ones_string(predictions_str)

    return jsonify({
        "message": predictions_str,
        "stats": {
            "total": total,
            "longest_fog": longest_fog
        }
    }), 200


def find_longest_consecutive_ones_string(binary_str):
    """
    从二进制字符串中查找最长连续 1 的长度

    参数:
        binary_str: 二进制字符串，如 '10111001'

    返回:
        最长连续 1 的长度
    """
    # 使用 '0' 分割字符串，过滤掉空字符串，然后计算最长子串的长度
    return max([len(group) for group in binary_str.split('0') if group], default=0)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
