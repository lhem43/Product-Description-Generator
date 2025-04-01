# Product Description Generator

- Author: Le Do Minh Anh
- Version: 1.0
- Repository link: [Product Description Generator](https://github.com/lhem43/Product-Description-Generator)

---
## 1. Introduction

Người dùng có thể cung cấp một đoạn tin nhắn bao gồm tên sản phẩm, thông số kỹ thuật, giá cả, các điểm đặc biệt,... và hệ thống sẽ sinh ra mô tả tương ứng để bán sản phẩm đó trên các sàn thương mại điện tử.

## 2. Installation

Để chạy được project này, bạn cần clone dự án về, cài đặt đầy đủ các thư viện trong requirements.txt và request file model fine-tuned-bert từ chúng tôi để có thể sử dụng.

Sau khi có đầy đủ các file, tiến hành `cd` tới thư mục chứa dự án và thực hiện lệnh sau để khởi chạy server AI.

```bash
uvicorn server:app
```

Tiếp theo, `cd` tới thư mục web_server/fe_server để khởi chạy frontend bằng lệnh

```bash
npm i
node server.js
```

Sau khi hoàn thành, ứng dụng sẽ khởi chạy ở cổng 2222 và việc setup hoàn tất.

## Usage

Người dùng tiến hành nhập yêu cầu của mình vào ô `Enter your requirements...` và nhấp **Submit**.

Câu trả lời sẽ được trả về ở hộp **Output**.