import xml.etree.ElementTree as ET

# 把这里换成你的 XML 路径
XML_FILE = r'E:\Cardio_Data\Data18\18XML\annotations.xml'


def diagnose_xml(xml_path):
    print(f"🕵️‍♂️ 正在诊断文件: {xml_path} ...")
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 1. 检查是不是 CVAT for Images 格式
    images = root.findall('image')
    print(f"📊 发现 <image> 标签数量: {len(images)}")

    if len(images) == 0:
        print("⚠️ 警告: 没有找到 <image> 标签！")
        # 检查是不是 Video 格式
        tracks = root.findall('track')
        if len(tracks) > 0:
            print(f"💡 发现 {len(tracks)} 个 <track> 标签。你可能导出了 'CVAT for Video' 格式，而不是 'CVAT for Images'。")
        return

    # 2. 抽样检查前几张图
    print("\n🔍 详细扫描前 5 张有标注的图片：")
    found_labels = set()
    found_shapes = set()

    count = 0
    for image in images:
        # 查找所有子标签
        children = list(image)
        if len(children) == 0:
            continue

        count += 1
        if count > 5: break

        print(f"\n--- 图片: {image.get('name')} ---")
        for child in children:
            tag = child.tag
            label = child.get('label')
            found_shapes.add(tag)
            if label:
                found_labels.add(label)
            print(f"   发现形状: <{tag}>, 标签名称: '{label}'")

    print("\n" + "=" * 30)
    print("📋 诊断总结:")
    print(f"1. 你的 XML 里包含的标签名称有: {found_labels}")
    print(f"2. 你的 XML 里包含的形状类型有: {found_shapes}")

    if 'polygon' not in found_shapes:
        print("❌ 致命问题: 没有找到 <polygon>！脚本无法工作。")
    else:
        print("✅ 形状检查通过 (包含 polygon)。")

    print("💡 请根据上面的'标签名称'修改你的 python 脚本里的 CLASS_MAP。")


if __name__ == "__main__":
    diagnose_xml(XML_FILE)