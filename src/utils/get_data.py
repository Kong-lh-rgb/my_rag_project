import requests
from playwright.sync_api import sync_playwright
import os
import time

# 定义下载目录
DOWNLOAD_DIR = r"D:\pyproject\PythonProject3\data\origin"
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)


def download_annual_reports(stock_name, year):
    """
    下载指定公司指定年份的年度报告
    """
    base_url = "http://www.cninfo.com.cn/"
    search_url = f"{base_url}new/disclosure/stock"

    with sync_playwright() as p:
        # 启动浏览器（调试时设为可见模式）
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            accept_downloads=True,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = context.new_page()

        try:
            # 第一步：搜索操作
            print(f"正在搜索'{stock_name}{year}年年报'...")
            page.goto(search_url, wait_until="networkidle", timeout=60000)
            search_box = page.wait_for_selector('input[placeholder*="代码"]', timeout=60000)
            search_box.fill(f"{stock_name}年报{year}")
            search_box.press("Enter")

            # 第二步：点击年报链接
            print("等待搜索结果...")
            link_selector = f'a[href*="detail"]:has(span.tileSecName-content:has-text("{year} 年报"))'
            page.wait_for_selector(link_selector, timeout=15000)

            # 使用更可靠的点击方式
            with page.expect_navigation():
                page.click(link_selector)
            print("成功进入详情页")

            # 第三步：点击下载按钮
            print("准备点击下载按钮...")

            # 使用更精确的 CSS 选择器来定位下载按钮
            # 我们定位带有 "el-button--primary" 和 "公告下载" 文本的按钮
            download_selector = 'button.el-button--primary:has-text("公告下载")'

            # 等待按钮可见并可以点击
            page.wait_for_selector(download_selector, timeout=10000)
            page.click(download_selector)
            # 使用 Playwright 的下载监听功能
            with page.expect_download() as download_info:
                # 使用 .click() 方法点击元素
                page.click(download_selector)

            # 获取下载文件
            download = download_info.value

            # 保存文件
            filename = f"{stock_name}_{year}_年报.pdf"
            filepath = os.path.join(DOWNLOAD_DIR, filename)
            download.save_as(filepath)
            print(f"下载成功: {filepath}")

        except Exception as e:
            print(f"发生错误: {str(e)}")

            # 调试信息
            print("当前页面URL:", page.url)
            print("页面标题:", page.title())

            # 检查按钮是否存在
            buttons = page.query_selector_all('button')
            print(f"找到 {len(buttons)} 个按钮:")
            for btn in buttons:
                print(" - 按钮文本:", btn.text_content())

        finally:
            time.sleep(1)
            browser.close()


# 示例调用
download_annual_reports("腾讯", "2024")
