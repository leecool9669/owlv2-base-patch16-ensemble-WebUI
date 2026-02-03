# -*- coding: utf-8 -*-
"""OWLv2 零样本目标检测 WebUI 演示（不加载真实模型权重）。"""
from __future__ import annotations

import gradio as gr


def fake_load_model():
    """模拟加载模型，实际不下载权重，仅用于界面演示。"""
    return "模型状态：owlv2-base-patch16-ensemble 已就绪（演示模式，未加载真实权重）"


def fake_detect(image, text_queries: str):
    """模拟零样本目标检测并返回可视化描述。"""
    if image is None:
        return "请上传一张图片。", None
    queries = [s.strip() for s in (text_queries or "").strip().split("\n") if s.strip()]
    if not queries:
        return "请输入至少一个文本查询（每行一个，如：a photo of a cat）。", None
    lines = [
        "[演示] 已对零样本目标检测进行计算（未加载真实模型）。",
        f"输入图像尺寸：{getattr(image, 'size', 'N/A')}",
        f"文本查询数量：{len(queries)}",
        "",
        "检测结果示例（占位）：",
    ]
    for i, q in enumerate(queries[:8]):
        lines.append(f"  - 「{q[:40]}{'...' if len(q) > 40 else ''}」 -> 置信度: 0.{9 - i % 10}xx, 边界框: [x1, y1, x2, y2]")
    lines.append("\n加载真实 owlv2-base-patch16-ensemble 后，将在此显示实际检测框与置信度。")
    return "\n".join(lines), image


def build_ui():
    with gr.Blocks(title="OWLv2 Zero-Shot Object Detection WebUI") as demo:
        gr.Markdown("## OWLv2 零样本目标检测 · WebUI 演示")
        gr.Markdown(
            "本界面以交互方式展示 owlv2-base-patch16-ensemble 零样本文本条件目标检测的典型使用流程，"
            "包括模型加载状态与图像—文本查询—检测结果的可视化展示。"
        )

        with gr.Row():
            load_btn = gr.Button("加载模型（演示）", variant="primary")
            status_box = gr.Textbox(label="模型状态", value="尚未加载", interactive=False)
        load_btn.click(fn=fake_load_model, outputs=status_box)

        with gr.Tabs():
            with gr.Tab("零样本目标检测"):
                gr.Markdown(
                    "上传一张图片，并在下方输入一个或多个文本查询（每行一个），模型将检测图像中与文本描述匹配的目标并输出边界框与置信度。"
                )
                image_inp = gr.Image(type="pil", label="输入图像")
                queries_inp = gr.Textbox(
                    label="文本查询（每行一个）",
                    placeholder="例如：\na photo of a cat\na photo of a dog\na person",
                    lines=5,
                )
                out_text = gr.Textbox(label="检测结果说明", lines=12, interactive=False)
                out_image = gr.Image(label="可视化结果（演示时显示原图）", type="pil")
                run_btn = gr.Button("执行检测（演示）")
                run_btn.click(
                    fn=fake_detect,
                    inputs=[image_inp, queries_inp],
                    outputs=[out_text, out_image],
                )

        gr.Markdown(
            "---\n*说明：当前为轻量级演示界面，未实际下载与加载 owlv2-base-patch16-ensemble 模型参数。*"
        )

    return demo


def main():
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
