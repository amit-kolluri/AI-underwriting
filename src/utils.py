import uuid
import json
from pdf2image import convert_from_path


def _pdf2images(pdf_path, file_name):
    images = convert_from_path(pdf_path)
    index = 1
    image_list = []
    for i, image in enumerate(images):
        image_dict = {}
        # img = page.to_image()
        image_name = "{}-{}".format(file_name, index)
        image_path = "data/images/{}.png".format(image_name)
        image_dict["image_name"] = image_name
        image_dict["image_path"] = image_path
        image_dict["image_number"] = index
        index += 1
        image.save(image_path, "PNG")
        image_list.append(image_dict)
    return image_list


def _save_pdf(uploaded_file):
    bytes_data = uploaded_file.read()
    file_name = str(uuid.uuid1().hex)
    print("file_name:", file_name)
    attachment_path = "data/attachments/{}.pdf".format(file_name)
    f = open(attachment_path, 'wb')
    f.write(bytes_data)
    f.close()
    return attachment_path, file_name


def _save_ocr_text(file_name, raw_text):
    file_path = "data/ocr/{}.txt".format(file_name)
    with open(file_path, "w") as file:
        file.write(raw_text)


def __empty_result():
    final_response = {"date_of_submittion": "", "contains": {}}
    email_body_dict = {"body_text": "", "subject": "", "sender": [""],
                        "signature": "", "send_to": ["send_to_1", ""],
                       "send_cc": ["", ""], "send_bcc": ["", ""]}
    final_response["contains"]["body"] = email_body_dict
    attachments = []
    final_response["contains"]["attachments"] = attachments
    return final_response


def _save_results(file_name, dictionary):
    final_response = __empty_result()
    final_response["contains"]["attachments"].append(dictionary)
    result_path = "data/results/{}.json".format(file_name)
    with open(result_path, "w") as outfile:
        json.dump(final_response, outfile, indent=4)
    return final_response