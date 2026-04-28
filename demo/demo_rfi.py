# Copyright (c) Opendatalab. All rights reserved.
import asyncio
import os
from pathlib import Path

from demo.demo import run_demo


def main() -> None:
    demo_dir = Path(__file__).resolve().parent

    # Input for RFI scenario.
    input_path = demo_dir / "rfi"
    output_dir = demo_dir / "api_output_rfi"

    # Leave None to auto-start local mineru-api.
    api_url = None

    # Force OCR flow to make sure OCR hook is exercised.
    backend = "pipeline"
    parse_method = "ocr"
    language = "vi"
    formula_enable = True
    table_enable = True
    server_url = None
    start_page_id = 0
    end_page_id = None

    # OCR hook settings for VietOCR provider.
    os.environ["MINERU_OCR_HOOK_ENABLE"] = "true"
    os.environ["MINERU_OCR_HOOK_LANGS"] = "vi"
    os.environ["MINERU_OCR_HOOK_PROVIDER"] = "vietocr"

    asyncio.run(
        run_demo(
            input_path=input_path,
            output_dir=output_dir,
            api_url=api_url,
            backend=backend,
            parse_method=parse_method,
            language=language,
            formula_enable=formula_enable,
            table_enable=table_enable,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
        )
    )


if __name__ == "__main__":
    main()
