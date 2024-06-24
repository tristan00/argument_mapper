import re
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import random
import json
import os

from common import addon_path, gecko_path, upvote_regex, no_user, comment_author_selector, view_source_class_name, \
    subreddit_list, webpage_save_path, post_link_save_loc, json_save_path
from dataset_utils import get_all_files
from models import ContentUnit

'''
1. scrape debate subreddits, debate transcripts....
2. Build graph db of conversation
3. Use topic modeling or llms to simplify, split and consolidate graphs into arguments
'''

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options


def get_driver():
    gecko_path = "/Users/tristandelforge/Documents/arguments/geckodriver"

    options = Options()
    options.headless = False

    service = Service(executable_path=gecko_path)

    driver = webdriver.Firefox(service=service, options=options)
    driver.install_addon(addon_path, temporary=True)
    return driver


def extract_upvotes(upvotes_text: str) -> int:
    match = re.search(upvote_regex, upvotes_text)  # Handles both positive and negative numbers
    return int(match.group()) if match else None


def extract_comments(post_link, comment_element, parent_id):
    author_elements = comment_element.find_elements(By.CSS_SELECTOR, comment_author_selector)
    user = author_elements[0].text if author_elements else no_user

    try:
        comment_text = comment_element.find_element(By.CLASS_NAME, 'md').get_attribute('outerHTML')
    except:
        comment_text = comment_element.find_element(By.CSS_SELECTOR, "div.md").get_attribute('outerHTML')
    date = comment_element.find_element(By.TAG_NAME, "time").get_attribute("title")
    upvote_elements = comment_element.find_elements(By.CSS_SELECTOR, "span.score.unvoted")
    upvotes = extract_upvotes(upvote_elements[0].text) if upvote_elements else None
    try:
        source_body = comment_element.find_element(By.CLASS_NAME, view_source_class_name).text
    except:
        source_body = None

    # source_text =
    subreddit = post_link.split('/')[4]
    try:
        user_tag = comment_element.find_element(By.CLASS_NAME, "flairrichtext flaircolordark flair ").text
    except:
        try:
            user_tag = comment_element.find_element(By.CLASS_NAME, "flair ").text
        except:
            user_tag = ''

    permalink = f"https://old.reddit.com{comment_element.get_attribute('data-permalink')}"

    comment_id = comment_element.get_attribute('data-fullname')
    if comment_id:

        comment_unit = ContentUnit(
            url=permalink,
            subreddit=subreddit,
            source_text=source_body,
            author_tag=user_tag,
            user=user,
            text=comment_text,
            date=date,
            upvotes=upvotes,
            id=comment_id.replace('t1_', ''),
            parent_id=parent_id.replace('t1_', ''),
            is_post=False
        )

        # Check for replies

        comment_id_replaced = comment_id.replace('t1_', '')

        parent_divs = comment_element.find_elements(By.CSS_SELECTOR, f"div[id='siteTable_t1_{comment_id_replaced}']")
        if parent_divs:
            child_comments = parent_divs[0].find_elements(By.XPATH, "./div[@data-type='comment']")
            for child_comment in child_comments:
                reply_unit = extract_comments(post_link, child_comment, comment_id)
                if reply_unit:
                    comment_unit.add_reply(reply_unit)

        return comment_unit


def scrape_comments_from_url(driver, post_link, post_id):
    parent_div = driver.find_element(By.CSS_SELECTOR, "div[id^='siteTable_t3_']")
    top_level_comments = parent_div.find_elements(By.XPATH, "./div[@data-type='comment']")

    # top_level_comments = driver.find_elements(By.CSS_SELECTOR, "div.comment")
    comments = []
    for top_comment in top_level_comments:
        comment_unit = extract_comments(post_link, top_comment, post_id)
        comments.append(comment_unit)
    return comments


def scrape_post(post_link, driver, save_directory, web_save_directory):
    import datetime

    print(f"Starting link: {post_link}, {datetime.datetime.now().isoformat()}")
    post_id = post_link.split('/')[-3]  # Extracting the post ID from the URL
    file_path = os.path.join(save_directory, f'{post_id}.json')
    webpage_path = os.path.join(web_save_directory, f'{post_id}.html')

    if os.path.exists(file_path) and not os.path.exists(webpage_path):
        print(f"File for post ID {post_id} already exists. Skipping.")
        return

    if os.path.exists(webpage_path) and False:  # Not working, keep false here
        driver.get(f"file://{webpage_path}")
        print(f'loaded page from cache: {post_link}')
    else:
        driver.get(f'{post_link}?limit=500')
        time.sleep(10)

        source_links = driver.find_elements(By.CSS_SELECTOR, "a.noCtrlF[data-text='source']")
        try:
            source_links[0].click()
        except:
            import traceback
            traceback.print_exc()
        time.sleep(5)

        with open(webpage_path, 'w') as f:
            f.write(driver.page_source)

    post_title = driver.find_element(By.CSS_SELECTOR, "a.title").text
    post_author_element = driver.find_element(By.CSS_SELECTOR, ".tagline a.author")
    post_author = post_author_element.text if post_author_element else no_user

    try:
        source_body = driver.find_element(By.CSS_SELECTOR, '.usertext-edit.viewSource').text
    except:
        import traceback
        traceback.print_exc()
        source_body = ''

    try:
        post_body = driver.find_element(By.CLASS_NAME, 'usertext warn-on-unload').get_attribute('outerHTML')

    except:
        try:
            post_body = driver.find_element(By.CLASS_NAME, 'usertext warn-on-unload').get_attribute('outerHTML')

        except:
            try:
                # post_body = driver.find_element(By.CSS_SELECTOR, "[id^='form-t3_']").text

                post_body = driver.find_element(By.CSS_SELECTOR, "[id^='form-t3_']").get_attribute('outerHTML')
            except:
                import traceback
                try:
                    post_body = driver.find_element(By.CSS_SELECTOR, "[id^='thing_t3_']").find_element(By.CLASS_NAME,
                                                                                                       "div.md").get_attribute(
                        'outerHTML')

                except:
                    traceback.print_exc()
                    return
    time_element = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.TAG_NAME, "time"))
    )
    post_date = time_element.get_attribute("title")
    post_upvotes = extract_upvotes(driver.find_element(By.CSS_SELECTOR, "span.score.unvoted").text)
    try:
        post_tag = driver.find_element(By.CLASS_NAME, "top-matter").find_element(By.CLASS_NAME,
                                                                                 "flairrichtext flaircolorlight linkflairlabel ").text
        print(1, post_tag)
    except:
        try:
            post_tag = driver.find_element(By.CLASS_NAME, "top-matter").find_element(By.CLASS_NAME,
                                                                                     'flairrichtext flaircolorlight linkflairlabel res-flairSearch').text
            print(1, post_tag)
        except:
            try:
                post_tag = driver.find_element(By.CLASS_NAME, "top-matter").find_element(By.CLASS_NAME,
                                                                                         'linkflairlabel').text
                print(1, post_tag)
            except:
                post_tag = None

    try:
        author_tag = driver.find_element(By.CLASS_NAME, "top-matter").find_element(By.CSS_SELECTOR, "span.flair").text
        print(2, author_tag)
    except:
        author_tag = ""

    post_unit = ContentUnit(
        url=post_link,
        subreddit=post_link.split('/')[4],
        source_text=source_body,
        tag=post_tag,
        author_tag=author_tag,
        user=post_author,
        text=post_body,
        date=post_date,
        upvotes=post_upvotes,
        id=post_id.replace('t1_', ''),
        parent_id=None,
        is_post=True,
        title=post_title,
    )

    comments = scrape_comments_from_url(driver, post_link, post_id)
    for comment in comments:
        post_unit.add_reply(comment)

    with open(file_path, 'w') as f:
        json.dump(post_unit.dict(), f)

    print(f"Saved post ID {post_id} to {file_path}")
    time.sleep(5)


def scrape_reddit(post_link_save_loc: str,
                  subreddit_list: list,
                  max_results_to_add: int = 10):
    try:
        with open(post_link_save_loc, 'r') as f:
            url_dict = json.load(f)
    except:
        url_dict = dict()

    driver = get_driver()
    time.sleep(2)

    urls = list()

    for s in subreddit_list:
        urls.append(f'https://old.reddit.com/r/{s}')

    for s in subreddit_list:
        urls.append(f'https://old.reddit.com/r/{s}/top/?sort=top&t=all')

    for s in subreddit_list:
        urls.append(f'https://old.reddit.com/r/{s}/controversial/?sort=top&t=all')

    urls = list(set(urls))
    random.shuffle(urls)
    urls = urls[:max_results_to_add]

    for i in urls:
        try:
            print(i, len(url_dict))
            driver.get(i)

            element = WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.ID, "siteTable"))
            )
            time.sleep(10)
            print()

            links = [link.get_attribute('href') for link in element.find_elements(By.TAG_NAME, "a") if
                     link.get_attribute('href') and
                     'comments' in link.get_attribute('href') and 'comments' in link.text and 'Can' not in link.text and
                     int(link.text.split()[0].replace(',', '')) >= 50]
            comments_dict = {link.split('/')[-3]: link for link in links if link}
            url_dict.update(comments_dict)
            with open(post_link_save_loc, 'w') as f:
                json.dump(url_dict, f)
            print('success', i)
        except:
            import traceback
            traceback.print_exc()
            print('failure', i)
            time.sleep(10)
        time.sleep(10)

    driver.quit()


def scrape_post_list(post_link_save_loc):
    scrape_reddit(post_link_save_loc, subreddit_list)


def scrape_posts(post_link_save_loc, cached_only=False):
    with open(post_link_save_loc, 'r') as f:
        url_dict = json.load(f)

    url_list = list(url_dict.values())
    random.shuffle(url_list)
    driver = get_driver()

    for post_link in url_list:
        try:
            post_id = post_link.split('/')[-3]
            post_id = post_id.replace('t1_', '')
            webpage_path = os.path.join(webpage_save_path, f'{post_id}.html')
            if os.path.exists(webpage_path) or not cached_only:
                scrape_post(post_link,
                            driver,
                            json_save_path,
                            webpage_save_path)
        except:
            import traceback
            traceback.print_exc()
        if not cached_only:
            time.sleep(random.randint(2, 5))


def show_trailing_whitespace(text):
    marked_text = repr(text + "␣")  # U+2423 Open Box symbol represents a space character visually
    print(marked_text)


def get_source_html_text_df():
    df = get_all_files()
    df = df[(~df['source_text'].isnull()) & (df['source_text'].astype(str).str.len() > 25)]
    df['source_text_char_rank'] = df['source_text'].str.len().rank(method='min', ascending=True)
    df['text_char_rank'] = df['text'].str.len().rank(method='min', ascending=True)

    def count_unique_non_alnum(s):
        return len(set(c for c in s if not c.isalnum()))

    df['source_text_nonalnum_rank'] = df['source_text'].apply(count_unique_non_alnum).rank(method='min',
                                                                                           ascending=False)
    df['text_nonalnum_rank'] = df['text'].apply(count_unique_non_alnum).rank(method='min', ascending=False)
    df = df[df['text_char_rank'] < 50]

    df['rank'] = (df['source_text_char_rank'] * df['text_char_rank'] * df['source_text_nonalnum_rank'] * df[
        'text_nonalnum_rank']).rank(method='min', ascending=True)
    df.sort_values('rank')


def main():
    scrape_post_list(post_link_save_loc)
    scrape_posts(post_link_save_loc)


if __name__ == '__main__':
    main()
