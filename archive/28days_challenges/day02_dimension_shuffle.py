import torch
# 引入我们昨天配置好的工具库
from utils.torch_playground import inspect, DEVICE

def experiment_permute_view():
    print("========= 🧪 Day 2: 维度变换与内存连续性实验 =========")
    
    # 1. 造数据：(Batch=2, Time=3, Channels=4)
    # 使用 arange 也就是有序数列，方便我们肉眼观察数据顺序的变化
    B, T, C = 2, 3, 4
    x = torch.arange(0, B*T*C).reshape(B, T, C).to(DEVICE)
    
    inspect(x, "原始数据 (B, T, C)")
    
    # ==========================================
    # 2. Permute: 维度换位 (B, T, C) -> (B, C, T)
    # 常见场景：把文本数据送入 1D-CNN 或 ResNet
    # ==========================================
    x_permuted = x.permute(0, 2, 1) #permute函数用于重新排列张量的维度顺序
    
    # 注意观察：Shape 变了，但是 stride (步长) 变得很奇怪，不再是连续的了
    inspect(x_permuted, "Permute 后 (B, C, T)")
    
    # ==========================================
    # 3. 💣 埋雷：尝试直接用 view 变形
    # ==========================================
    print("\n👉 尝试对 permute 后的张量直接使用 .view()...")
    try:
        # 试图把它展平，这在全连接层前很常见
        flatten_attempt = x_permuted.view(B, C * T)
    except RuntimeError as e:
        print(f"❌ 报错捕获成功！\n错误信息: {e}")
        print("💡 原因：Permute 只是改变了读取索引的顺序，内存中数据并没有真的搬家。")
        print("   但 view 要求内存必须是【连续】(Contiguous) 的。")

    # ==========================================
    # 4. ✅ 拆弹：contiguous() 的作用
    # ==========================================
    print("\n👉 使用 .contiguous() 修复...")
    x_contiguous = x_permuted.contiguous()
    
    inspect(x_contiguous, "Contiguous 后")
    
    # 现在 view 可以用了
    flatten_success = x_contiguous.view(B, C * T)
    print(f"✅ View 成功！Shape: {flatten_success.shape}")

    # ==========================================
    # 5. 还原：(B, C, T) -> (B, T, C)
    # ==========================================
    # 注意：还原也要用 permute 把维度换回去，不能用 view/reshape 硬捏
    x_restored = x_permuted.permute(0, 2, 1)
    
    # 验证数据是否和一开始一样
    is_same = torch.equal(x, x_restored)
    print(f"\n🔄 还原验证: {'完美一致 🎉' if is_same else '数据错乱 😱'}")

if __name__ == "__main__":
    experiment_permute_view()