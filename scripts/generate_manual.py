"""
生成《使用说明.pdf》— 面向非技术用户的运行指南
"""
from fpdf import FPDF
from pathlib import Path


class ChinesePDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        # 注册中文字体
        font_path = "C:/Windows/Fonts/simhei.ttf"
        self.add_font("SimHei", "", font_path)
        self.add_font("SimHei", "B", font_path)
        self.set_auto_page_break(True, 20)

    def header(self):
        if self.page_no() > 1:
            self.set_font("SimHei", "", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 6, "电信客户流失预测 — 使用说明", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("SimHei", "", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"第 {self.page_no()} 页", align="C")

    def title_block(self, text):
        self.set_font("SimHei", "B", 18)
        self.set_text_color(30, 60, 120)
        self.cell(0, 12, text, align="C")
        self.ln(16)

    def section_title(self, text):
        self.ln(4)
        self.set_x(self.l_margin)
        self.set_font("SimHei", "B", 13)
        self.set_text_color(30, 60, 120)
        self.cell(0, 10, text)
        self.ln(14)

    def body(self, text):
        self.set_x(self.l_margin)
        self.set_font("SimHei", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(self.w - self.l_margin - self.r_margin, 7, text)
        self.set_x(self.l_margin)

    def code_block(self, text):
        """灰色底色的命令块，结束后自动复位到左边距"""
        self.set_x(self.l_margin)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(60, 60, 60)
        self.set_font("SimHei", "", 9)
        block_width = self.w - self.l_margin - self.r_margin
        for line in text.strip().split("\n"):
            self.set_x(self.l_margin)
            self.cell(block_width, 6, f"  {line}", fill=True)
            self.ln()
        self.ln(2)
        self.set_x(self.l_margin)

    def bullet(self, text, indent=10):
        """带缩进的列表项，结束后自动复位"""
        self.set_x(self.l_margin + indent)
        self.set_font("SimHei", "", 10)
        self.set_text_color(40, 40, 40)
        self.cell(5, 7, "-")
        bullet_width = self.w - self.l_margin - self.r_margin - indent - 5
        self.multi_cell(bullet_width, 7, text)
        self.set_x(self.l_margin)

    def note_box(self, text):
        """重要提示框，结束后自动复位"""
        self.set_x(self.l_margin)
        self.set_fill_color(255, 243, 205)
        self.set_draw_color(200, 180, 100)
        self.set_text_color(100, 80, 20)
        self.set_font("SimHei", "", 9)
        box_width = self.w - self.l_margin - self.r_margin
        self.multi_cell(box_width, 6, f"[注意] {text}", fill=True)
        self.set_x(self.l_margin)


def generate_pdf():
    pdf = ChinesePDF()
    project_root = Path(__file__).resolve().parent.parent

    # ===== 第一章：项目简介 =====
    pdf.add_page()
    pdf.title_block("一、项目简介")
    pdf.set_fill_color(230, 240, 255)
    pdf.set_text_color(30, 60, 120)
    pdf.set_font("SimHei", "B", 10)
    pdf.set_x(18)
    pdf.cell(174, 8, "  版本：v0.2 改进版  |  EDA + SMOTE + GridSearchCV + 消融实验 + 46 单元测试", fill=True)
    pdf.ln(14)
    pdf.body(
        "本项目使用机器学习方法，基于电信客户的基本信息、消费行为和服务使用情况，"
        "预测客户是否存在流失风险。项目对比了三种常用算法（逻辑回归、随机森林、XGBoost），"
        "通过 SMOTE 处理数据不均衡，GridSearchCV 超参数调优，并输出可视化图表帮助理解分析结果。"
    )
    pdf.section_title("当前进度")
    pdf.body(
        "数据获取 已完成 | EDA 分析 已完成 | 数据清洗 已完成 | 特征工程 已完成\n"
        "SMOTE 过采样 已完成 | GridSearchCV 调优 已完成 | 消融实验 已完成\n"
        "单元测试 已完成 (46 tests) | 论文撰写 待做 | PPT 制作 待做"
    )
    pdf.section_title("数据集说明")
    pdf.body(
        "数据来源：Kaggle 公开数据集「Telco Customer Churn」\n"
        "数据规模：7,043 条客户记录，包含 20 个特征字段\n"
        "特征包括：性别、年龄、是否退休、入网时长、服务订阅情况、合同类型、"
        "支付方式、月费用、总费用等\n"
        "预测目标：客户是否流失（Churn：是/否）"
    )
    pdf.section_title("你将得到什么")
    pdf.bullet("模型对比表：三种算法的准确率、召回率、F1 分数、AUC")
    pdf.bullet("EDA 探索分析图：数据分布、类别特征流失率、相关性热力图")
    pdf.bullet("客户流失分布图（饼图）")
    pdf.bullet("特征重要性排序图（哪些因素最影响流失）")
    pdf.bullet("ROC 曲线对比图（模型性能可视化）")
    pdf.bullet("混淆矩阵图（预测正确/错误的数量分布）")
    pdf.bullet("消融实验对比表（各模块贡献量化）")

    # ===== 第二章：环境准备 =====
    pdf.add_page()
    pdf.title_block("二、环境准备（只需做一次）")

    pdf.section_title("2.1 安装 Python")
    pdf.body(
        "本项目需要 Python 3.8 或更高版本。如果电脑上已有 Python，可跳过此步骤。"
    )
    pdf.bullet("打开浏览器，访问：https://www.python.org/downloads/")
    pdf.bullet('点击黄色按钮 "Download Python" 下载安装包')
    pdf.bullet("运行下载的 .exe 文件，安装时务必勾选 [x] Add Python to PATH")
    pdf.bullet("点击 Install Now，等待安装完成")
    pdf.note_box("安装时一定要勾选「Add Python to PATH」，否则后续命令行无法识别 python 命令。")

    pdf.section_title("2.2 如何进入项目目录（重要）")
    pdf.body(
        "拿到项目文件夹后，首先需要让命令行切换到该目录下。根据你的存放位置，"
        "选择以下任意一种方法："
    )
    pdf.set_font("SimHei", "B", 10)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 8, "方法一：拖拽文件夹（推荐，最简单）")
    pdf.ln(10)
    pdf.set_font("SimHei", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.bullet("按 Win+R，输入 cmd，回车打开命令行窗口")
    pdf.bullet("输入 cd 加一个空格（注意空格！）：")
    pdf.code_block("cd ")
    pdf.bullet("打开文件资源管理器，找到你存放的项目文件夹")
    pdf.bullet("用鼠标将整个项目文件夹拖入命令行窗口")
    pdf.bullet("此时命令行会自动填上文件夹的完整路径，例如：")
    pdf.code_block('cd C:\\Users\\你的用户名\\Desktop\\医疗大数据综合应用实验')
    pdf.bullet("按回车，即进入项目目录")

    pdf.set_font("SimHei", "B", 10)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 8, "方法二：右键打开（Win11 推荐）")
    pdf.ln(10)
    pdf.set_font("SimHei", "", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.bullet("在文件资源管理器中，打开项目文件夹")
    pdf.bullet("点击地址栏，输入 cmd，回车")
    pdf.bullet("命令行窗口将自动定位到当前目录")
    pdf.note_box("进入项目目录后，后续所有命令都在该窗口执行，不要关闭。你可以用 dir 命令确认当前目录下是否有 src、data 等文件夹。")

    pdf.section_title("2.3 验证 Python 安装")
    pdf.body("在命令行窗口中输入以下命令并回车：")
    pdf.code_block("python --version")
    pdf.bullet('如果显示 "Python 3.x.x" 则表示安装成功')

    pdf.section_title("2.4 安装项目依赖")
    pdf.body("确保命令行已进入项目目录（参考 2.2 节），然后执行：")
    pdf.code_block("pip install -r requirements.txt")
    pdf.body("等待安装完成（约 2-5 分钟，因网络速度而异）。看到 Successfully installed 即表示成功。")
    pdf.note_box('如果安装速度很慢，可以试试：pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple')
    pdf.note_box('本项目已自带中文字体（assets/fonts/NotoSansSC.ttf），图表中文标注在 Windows/Mac/Linux 下均可正常显示，无需额外安装字体。')

    # ===== 第三章：运行项目 =====
    pdf.add_page()
    pdf.title_block("三、运行项目")

    pdf.section_title("3.1 准备数据")
    pdf.body("数据集文件需要提前下载并放置到指定目录。（通常已由项目负责人准备好）")
    pdf.bullet("确认 data/raw/ 目录下存在 WA_Fn-UseC_-Telco-Customer-Churn.csv 文件")
    pdf.bullet("如果缺少该文件，请从以下地址下载：")
    pdf.code_block("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    pdf.bullet('下载后解压，将 .csv 文件放入 data/raw/ 目录')

    pdf.section_title("3.2 一键运行")
    pdf.body("确保命令行已进入项目目录（参考 2.2 节），然后执行：")
    pdf.code_block("python src/main.py")
    pdf.body("程序运行后将自动完成以下步骤：")
    pdf.bullet("加载并清洗数据（处理缺失值）")
    pdf.bullet("特征工程（编码、标准化、划分训练/测试集）")
    pdf.bullet("训练三个模型（逻辑回归、随机森林、XGBoost）")
    pdf.bullet("交叉验证评估模型稳定性")
    pdf.bullet("在测试集上对比模型性能")
    pdf.bullet("生成 4 张可视化图表到 results/ 目录")
    pdf.body("整个流程约 10-30 秒完成。")

    # ===== 第四章：结果说明 =====
    pdf.add_page()
    pdf.title_block("四、结果说明")

    pdf.section_title("4.1 控制台输出")
    pdf.body(
        "程序运行过程中，命令行窗口会实时显示处理进度和模型评估结果。"
        "重点关注最后的模型对比表："
    )
    pdf.code_block(
        "Model                  Accuracy  Precision  Recall  F1     AUC\n"
        "---------------------------------------------------------------\n"
        "LogisticRegression      0.8048    0.6581     ...    ...   0.8426\n"
        "RandomForest            0.7906    0.6561     ...    ...   0.8429\n"
        "XGBoost                 0.7935    0.6379     ...    ...   0.8363\n"
        "---------------------------------------------------------------\n"
        "最佳模型: RandomForest (AUC=0.8429)"
    )
    pdf.body("各项指标含义：")
    pdf.bullet("Accuracy（准确率）：预测正确的比例，越高越好")
    pdf.bullet("Precision（精确率）：预测为流失的客户中，真正流失的比例")
    pdf.bullet("Recall（召回率）：真正流失的客户中，被模型找出的比例")
    pdf.bullet("F1：精确率和召回率的调和平均，综合衡量模型效果")
    pdf.bullet("AUC：模型区分流失/非流失客户的能力，越接近 1 越好")

    pdf.section_title("4.2 可视化图表（results/ 目录）")
    pdf.body("以下 4 张图片在程序运行后自动生成，可直接双击打开查看：")

    pdf.set_font("SimHei", "B", 10)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, "① churn_distribution.png — 客户流失分布")
    pdf.ln(10)
    pdf.set_font("SimHei", "", 10)
    pdf.body("饼图展示数据集中流失客户与留存客户的比例，直观了解数据失衡程度。")

    pdf.set_font("SimHei", "B", 10)
    pdf.cell(0, 8, "② feature_importance.png — 特征重要性排序")
    pdf.ln(10)
    pdf.set_font("SimHei", "", 10)
    pdf.body("横向条形图，展示哪些因素对预测客户流失最重要。排名越靠前，影响越大。")

    pdf.set_font("SimHei", "B", 10)
    pdf.cell(0, 8, "③ roc_curves.png — ROC 曲线对比")
    pdf.ln(10)
    pdf.set_font("SimHei", "", 10)
    pdf.body("三条曲线分别对应三种模型。曲线越靠近左上角，模型性能越好。")
    pdf.body("对角线（虚线）代表随机猜测的基线，AUC 值标注在图例中。")

    pdf.set_font("SimHei", "B", 10)
    pdf.cell(0, 8, "④ confusion_matrices.png — 混淆矩阵")
    pdf.ln(10)
    pdf.set_font("SimHei", "", 10)
    pdf.body("每个模型一个热力图，展示预测结果与真实结果的交叉统计。")
    pdf.body("左上=正确预测未流失；右下=正确预测流失；右上=误报；左下=漏报。")

    # ===== 第五章：常见问题 =====
    pdf.add_page()
    pdf.title_block("五、常见问题")

    qa = [
        (
            'Q1：提示 "python 不是内部或外部命令"',
            "Python 未正确安装或未添加到 PATH。请重新运行 Python 安装包，"
            "选择「Modify」，确保勾选「Add Python to environment variables」，"
            "然后重启命令行窗口。",
        ),
        (
            "Q2：pip install 报错或连接超时",
            "网络问题导致无法从默认源下载。使用国内镜像源重试：\n"
            "pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple",
        ),
        (
            "Q3：提示找不到数据文件",
            "请确认 data/raw/ 目录下存在 WA_Fn-UseC_-Telco-Customer-Churn.csv 文件。"
            "文件名必须完全一致（含大小写），不能是中文名或带有 (1) 等后缀。",
        ),
        (
            "Q4：运行时出现红色的错误信息",
            "截图保存错误信息，联系项目工程负责人排查。常见原因：Python 版本太低（需 ≥3.8）、"
            "依赖包未完整安装、数据文件损坏。",
        ),
        (
            "Q5：生成的图片中文显示为方框",
            "项目已配置使用系统自带的中文字体。如果出现方框，"
            "通常是因为系统字体版本问题，可联系项目负责人处理。",
        ),
        (
            "Q6：想修改模型参数或重新训练",
            "编辑 configs/config.yaml 文件（可用记事本打开），修改相关参数后重新运行即可。"
            "建议在修改前备份原文件。",
        ),
    ]

    for q, a in qa:
        pdf.set_font("SimHei", "B", 10)
        pdf.set_text_color(30, 60, 120)
        pdf.cell(0, 8, q)
        pdf.ln(10)
        pdf.set_font("SimHei", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 7, a)
        pdf.ln(4)

    # ===== 附录 =====
    pdf.section_title("附录：项目文件结构")
    pdf.set_font("SimHei", "", 9)
    pdf.set_text_color(60, 60, 60)
    structure = (
        "项目根目录/\n"
        "├── README.md               项目说明文档\n"
        "├── 使用说明.pdf             本文件（操作指南）\n"
        "├── requirements.txt         Python 依赖列表\n"
        "├── configs/config.yaml      配置文件（随机种子、模型参数）\n"
        "├── src/main.py              程序入口（运行此文件即可）\n"
        "├── data/raw/                原始数据存放目录\n"
        "├── results/                 输出结果（图片）\n"
        "└── notebooks/               Jupyter 探索笔记（进阶用户）"
    )
    for line in structure.strip().split("\n"):
        pdf.set_x(18)
        pdf.cell(170, 5.5, line)
        pdf.ln()

    # 保存
    output_path = project_root / "使用说明.pdf"
    pdf.output(str(output_path))
    print(f"PDF 已生成: {output_path}")


if __name__ == "__main__":
    generate_pdf()
