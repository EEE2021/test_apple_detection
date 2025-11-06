import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def load_image(image_path):
    """加载图像并转换为RGB格式"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"加载图像时出错: {e}")
        return None

def preprocess_image(image):
    """图像预处理：高斯模糊减少噪声"""
    return cv2.GaussianBlur(image, (5, 5), 0)

def detect_red_apples(image):
    """检测红色苹果"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # 红色在HSV空间中不连续，需要两个范围
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    return red_mask

def detect_green_apples(image):
    """检测绿色苹果"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # 定义绿色在HSV空间中的范围
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    return green_mask

def apply_morphology(mask):
    """应用形态学操作去除噪声"""
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    
    # 开运算：先腐蚀后膨胀，去除小物体
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 闭运算：先膨胀后腐蚀，填充内部孔洞
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return closing

def find_and_filter_contours(mask, min_area=1000):
    """查找轮廓并过滤小面积区域"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 过滤小面积轮廓
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            filtered_contours.append(contour)
    
    return filtered_contours

def draw_contours_and_boxes(image, contours, color, label):
    """在图像上绘制轮廓和边界框"""
    result = image.copy()
    centers = []
    count = 0
    
    for contour in contours:
        # 绘制轮廓
        cv2.drawContours(result, [contour], -1, color, 2)
        
        # 计算边界框
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        
        # 计算中心点
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
            
            # 在中心点绘制标签
            cv2.putText(result, f"{label} {count+1}", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            count += 1
    
    return result, count

def detect_and_classify_apples(image_path, min_area=1000):
    """检测并分类苹果"""
    # 加载图像
    original_image = load_image(image_path)
    if original_image is None:
        return None, None, None
    
    # 预处理图像
    processed_image = preprocess_image(original_image)
    
    # 检测红色和绿色苹果
    red_mask = detect_red_apples(processed_image)
    green_mask = detect_green_apples(processed_image)
    
    # 形态学处理
    red_mask_clean = apply_morphology(red_mask)
    green_mask_clean = apply_morphology(green_mask)
    
    # 查找并过滤轮廓
    red_contours = find_and_filter_contours(red_mask_clean, min_area)
    green_contours = find_and_filter_contours(green_mask_clean, min_area)
    
    # 绘制结果
    result_image, red_count = draw_contours_and_boxes(
        original_image, red_contours, (255, 0, 0), "Red"
    )
    result_image, green_count = draw_contours_and_boxes(
        result_image, green_contours, (0, 255, 0), "Green"
    )
    
    return result_image, red_count, green_count

def visualize_results(original_image, result_image, red_count, green_count):
    """可视化检测结果"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    # 显示原始图像
    axes[0].imshow(original_image)
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 显示检测结果
    axes[1].imshow(result_image)
    axes[1].set_title(f'检测结果 - 红色: {red_count}, 绿色: {green_count}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 示例图像路径，替换为你自己的图像
    image_path = "apples.jpg"
    
    # 检测苹果
    result_image, red_count, green_count = detect_and_classify_apples(image_path)
    
    if result_image is not None:
        # 可视化结果
        original_image = load_image(image_path)
        visualize_results(original_image, result_image, red_count, green_count)
        
        # 打印统计结果
        print(f"检测到 {red_count} 个红色苹果和 {green_count} 个绿色苹果。")
    else:
        print("苹果检测失败，请检查图像路径或图像格式。")