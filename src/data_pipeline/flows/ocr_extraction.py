from prefect import task


@task
def ocr_extraction() -> str:
    extracted_text = "Politechnika Wrocławska is located in Wrocław, Poland. \
    It is one of the top technical universities in the country."
    return extracted_text
