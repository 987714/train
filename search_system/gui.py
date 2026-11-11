import sys

import random
import time
import numpy


from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTextEdit, QSlider, QFileDialog,
    QGroupBox, QGridLayout, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt
from search_func import Search  # 请确保此模块存在并实现Search类

class PaddleNLPSearchUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(300, 100, 1000, 600)
        self.setStyleSheet(self.load_qss())
        self.result_count = 0  # 记录结果项数量，用于序号计算
        self.init_ui()

    def load_qss(self):   #  创建 qss 样式表
        """完整界面样式（含时间显示、按钮交互等）"""
        return """
        
        QWidget {
            background-color: #fafafa;
            font-family: 'Microsoft YaHei';
            font-size: 14px;
        }
        QLabel {
            color: #333;
        }
        QGroupBox {
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-top: 10px;
            padding: 15px;
            font-weight: bold;
        }
        QLineEdit {
            height: 36px;
            border: 1px solid #ccc;
            border-radius: 6px;
            padding-left: 10px;
        }
        QLineEdit:focus {
            border: 1px solid #ff4d4f;
        }
        /* 时间显示框样式 */
        QLineEdit#timeDisplay {
            background-color: #f0f0f0;
            color: #333;
            font-weight: normal;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding-left: 10px;
        }
        QPushButton {
            height: 36px;
            border-radius: 6px;
            padding: 0 16px;
            font-weight: bold;
            background-color: #ff4d4f;
            color: white;
            border: none;
        }
        QPushButton:hover {
            background-color: #ff6666;
        }
        QPushButton:pressed {
            background-color: #e63946;
        }
        QProgressBar {
            height: 14px;
            border: 1px solid #ccc;
            border-radius: 7px;
            background-color: #f5f5f5;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #4caf50;
            border-radius: 7px;
        }
        QTextEdit {
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: white;
            padding: 8px;
        }
        QScrollArea {
            border: none;
        }
        QScrollBar:vertical {
            width: 8px;
            background-color: #f0f0f0;
        }
        QScrollBar::handle:vertical {
            background-color: #ccc;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #bbb;
        }
        """

    def init_ui(self):
        """初始化界面布局"""
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)  # 左右面板间距
        main_layout.setContentsMargins(20, 20, 20, 20)  # 窗口边距

        # ==================== 左侧选项面板 ====================
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)
        left_panel.setContentsMargins(0, 0, 0, 0)

        # 选项组框
        option_box = QGroupBox("选项")
        option_box.setFlat(False)  # 保留组框立体效果
        option_layout = QGridLayout()
        option_layout.setHorizontalSpacing(15)
        option_layout.setVerticalSpacing(10)
        option_layout.setContentsMargins(15, 15, 15, 15)  # 组框内边距

        # 1. 最大召回数量滑块
        option_layout.addWidget(QLabel("最大召回数量"), 0, 0)
        self.answer_slider = QSlider(Qt.Horizontal)
        self.answer_slider.setRange(1, 50)
        self.answer_slider.setValue(3)
        self.answer_label = QLabel(str(self.answer_slider.value()))
        self.answer_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        option_layout.addWidget(self.answer_slider, 0, 1)
        option_layout.addWidget(self.answer_label, 0, 2)

        # 2. 最大检索数量滑块
        option_layout.addWidget(QLabel("最大检索数量"), 1, 0)
        self.search_slider = QSlider(Qt.Horizontal)
        self.search_slider.setRange(1, 50)
        self.search_slider.setValue(30)
        self.search_label = QLabel(str(self.search_slider.value()))
        self.search_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        option_layout.addWidget(self.search_slider, 1, 1)
        option_layout.addWidget(self.search_label, 1, 2)

        # 3. 本次查询用时显示
        option_layout.addWidget(QLabel("本次查询用时："), 2, 0)
        self.time_display = QLineEdit()
        self.time_display.setObjectName("timeDisplay")  # 关联样式
        self.time_display.setReadOnly(True)
        self.time_display.setText("0.0000秒")
        option_layout.addWidget(self.time_display, 2, 1, 1, 2)  # 跨2列

        # 4. 文件上传按钮
        option_layout.addWidget(QLabel("文件上传："), 3, 0)
        upload_btn = QPushButton("📁 选择文件")
        upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                color: #333;
                font-weight: normal;
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        upload_btn.setCursor(Qt.PointingHandCursor)
        upload_btn.clicked.connect(self.select_file)
        option_layout.addWidget(upload_btn, 3, 1, 1, 2)

        option_box.setLayout(option_layout)
        left_panel.addWidget(option_box)
        left_panel.addStretch()  # 占位让组框靠上

        # ==================== 右侧主内容面板 ====================
        right_panel = QVBoxLayout()
        right_panel.setSpacing(20)
        right_panel.setContentsMargins(0, 0, 0, 0)

        # 查询输入区
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("请输入查询文献内容，例如：生态系统分类方案研发")
        self.query_input.setStyleSheet("""
            QLineEdit {
                flex: 1;
                height: 36px;
                border: 1px solid #ccc;
                border-radius: 6px;
                padding-left: 10px;
            }
            QLineEdit:focus {
                border: 1px solid #ff4d4f;
            }
        """)
        self.query_input.setMinimumWidth(300)
        input_layout.addWidget(self.query_input)

        # 运行按钮
        run_btn = QPushButton("运行")
        run_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4d4f;
                color: white;
                font-weight: bold;
                border-radius: 6px;
                padding: 0 24px;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
            QPushButton:pressed {
                background-color: #e63946;
            }
        """)
        run_btn.setCursor(Qt.PointingHandCursor)
        run_btn.clicked.connect(self.run_search)
        input_layout.addWidget(run_btn)

        # 随机生成按钮
        random_btn = QPushButton("随机生成")
        random_btn.setStyleSheet("""
            QPushButton {
                background-color: #666;
                color: white;
                font-weight: bold;
                border-radius: 6px;
                padding: 0 24px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #444;
            }
        """)
        random_btn.setCursor(Qt.PointingHandCursor)
        random_btn.clicked.connect(self.random_query)
        input_layout.addWidget(random_btn)

        right_panel.addLayout(input_layout)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #ddd;")
        right_panel.addWidget(line)

        # 结果标题
        result_title = QLabel("<b>返回结果：</b>")
        result_title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px 0;")
        right_panel.addWidget(result_title)

        # 结果滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                width: 8px;
                background-color: #f0f0f0;
            }
            QScrollBar::handle:vertical {
                background-color: #ccc;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #bbb;
            }
        """)

        # 结果容器（用于放置所有结果项）
        self.result_container = QWidget()
        self.result_layout = QVBoxLayout(self.result_container)
        self.result_layout.setSpacing(10)
        self.result_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setWidget(self.result_container)
        right_panel.addWidget(self.scroll_area)

        # 绑定滑块信号
        self.answer_slider.valueChanged.connect(lambda v: self.answer_label.setText(str(v)))
        self.search_slider.valueChanged.connect(lambda v: self.search_label.setText(str(v)))

        # 添加左右面板到主布局
        main_layout.addLayout(left_panel, 2)  # 左侧占2份宽度
        main_layout.addLayout(right_panel, 5)  # 右侧占5份宽度

        self.setLayout(main_layout)

    def select_file(self):
        """选择文献文件"""
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "选择文献文件",
            "",
            "文本文件 (*.txt);;Word文件 (*.docx);;PDF文件 (*.pdf);;所有文件 (*)"
        )
        if fname:
            self.show_message(f"✅ 已选择文件：{fname}")

    def show_message(self, text):
        """显示提示信息（灰色小字）"""
        msg_label = QLabel(text)
        msg_label.setStyleSheet("color: gray; font-size: 12px; margin: 5px 0;")
        self.result_layout.addWidget(msg_label)

    def run_search(self):
        """执行搜索逻辑"""
        start_time = time.time()  # 记录查询开始时间

        # 1. 校验查询内容
        query = self.query_input.text().strip()
        if not query:
            self.show_message("⚠️ 请输入查询内容后再运行。")
            return

        # 2. 清空旧结果和提示
        for i in reversed(range(self.result_layout.count())):
            widget = self.result_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.result_count = 0  # 重置为0，准备重新计数结果项
        # 3. 获取滑块参数
        num_recall = self.answer_slider.value()  # 最大召回数量
        num_search = self.search_slider.value()  # 最大展示数量

        try:
            # 4. 调用搜索函数（需确保search_func.Search已实现）
            rank_result = Search(query, num_recall)
            cnt = 0
            for item in rank_result:
                text = item.get('title', '无标题')  # 兼容无title字段的情况
                score = item.get('pred_prob', 0.0)  # 兼容无score字段的情况
                self.add_result(text, score)
                cnt += 1
                if cnt >= num_search:
                    break
            self.show_message(f"✅ 搜索完成，共找到 {cnt} 条结果")

        except Exception as e:
            self.show_message(f"❌ 搜索失败：{str(e)}")

        # 5. 更新查询用时
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.time_display.setText("{:.4f}秒".format(elapsed_time))

        # 6. 清空输入框（为下次查询准备）
        self.query_input.clear()

    def add_result(self, text, score):
        """添加单条结果到列表（序号+文本+分数，Tab分隔）"""
        # 结果项容器（带 hover 效果）
        container = QWidget()
        container.setStyleSheet("""
            QWidget {
                background-color: #fff;
                border: 1px solid #eee;
                border-radius: 8px;
                padding: 12px 15px;
                margin-bottom: 8px;
                transition: all 0.3s ease;
            }
            QWidget:hover {
                border-color: #ff4d4f;
                box-shadow: 0 2px 4px rgba(255,77,79,0.1);
            }
        """)

        # 计算序号（自动从1开始，因result_layout已清空）
        result_index = self.result_count + 1
        self.result_count += 1
        # 格式化内容：序号 + 文本 + 分数（保留4位小数）
        result_content = f"{result_index}\t{text}\t{score:.4f}"
        # 等宽字体确保Tab对齐
        label = QLabel(result_content)
        label.setStyleSheet("""
            QLabel {
                font-family: 'Courier New', monospace;
                font-size: 14px;
                color: #333;
                line-height: 1.5;
            }
        """)
        label.setContentsMargins(0, 0, 0, 0)  # 去除label内边距

        # 容器布局（包裹label）
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label)

        # 添加到结果列表
        self.result_layout.addWidget(container)

    def random_query(self):  # 随机生成按钮 槽函数
        """随机生成查询内容"""
        random_queries = [
            "深度学习在自然语言处理中的应用",
            "气候变化对生物多样性的影响",
            "新冠疫情下的公共卫生政策分析",
            "人工智能伦理问题研究",
            "新能源汽车技术发展趋势",
            "乡村振兴背景下的农业现代化路径",
            "元宇宙技术的现状与未来展望"
        ]
        self.query_input.setText(random.choice(random_queries))


if __name__ == '__main__':  # 用户双击运行脚本 → __name__设为 "__main__"
    app = QApplication(sys.argv) # 创建Qt应用程序实例，sys.argv用于接收命令行参数
    app.setStyle("Fusion")  # 使用Fusion样式统一外观
    window = PaddleNLPSearchUI() # 实例化自定义的主窗口类 PaddleNLPSearchUI
    window.show()   # 显示主窗口
    # 启动应用程序的事件循环
    sys.exit(app.exec_())  # app.exec_() 进入主循环，直到窗口关闭； sys.exit() 确保程序退出时返回正确的状态码

