from pymilvus import connections, utility

# Kết nối Milvus
connections.connect("default", host="localhost", port="19530")

# Xóa collection cũ bị lệch dim (thường tên là keyframes hoặc video_keyframes)
collections = utility.list_collections()
print(f"Các collections hiện có: {collections}")

for col_name in collections:
    utility.drop_collection(col_name)
    print(f"✅ Đã xóa collection: {col_name}")

print("Vui lòng restart lại API để hệ thống tự tạo Collection mới với đúng dim!")