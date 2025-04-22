import argparse
from bs4 import BeautifulSoup
import markdownify

def extract_and_convert(html_path, md_path):
    try:
        with open(html_path, "r", encoding="windows-1252") as f:
            soup = BeautifulSoup(f, "html.parser")

        target_header = soup.find("h2", string="AI Analysis of Visualizations")
        if target_header:
            content_div = target_header.find_parent("div", class_="section-header")
            result = []
            for sibling in content_div.find_next_siblings():
                if sibling.name == "div" and "section-header" in sibling.get("class", []):
                    break
                result.append(str(sibling))

            extracted_html = "\n".join(result)
            markdown_text = markdownify.markdownify(extracted_html, heading_style="ATX")

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)

            print(f"First 100 lines of the markdown file:")
            with open(md_path, "r", encoding="utf-8") as f:
                print("".join(f.readlines()[:100]))

            return md_path
        else:
            print("Target section not found in HTML.")
            return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and convert HTML to Markdown")
    parser.add_argument("--html", default="combined_pattern_analysis.html", help="Path to input HTML file")
    parser.add_argument("--md", default="data/complete_analysis.md", help="Path to output Markdown file")
    args = parser.parse_args()

    result_path = extract_and_convert(args.html, args.md)
    print(f"Markdown saved to: {result_path}")
