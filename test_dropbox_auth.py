import dropbox
from utils.config_loader import load_config

config = load_config()

ACCESS_TOKEN = config['dropbox']['access_token']

dbx = dropbox.Dropbox(ACCESS_TOKEN)

print(dbx.users_get_current_account())

for entry in dbx.files_list_folder('').entries:
    print(entry.name)

dbx.files_upload(b"Potential headline: Game 5 a nail-biter as Warriors inch out Cavs", '/cavs vs warriors/game 5/story.txt')

print(dbx.files_get_metadata('/Cavs vs Warriors/Game 5/story.txt').server_modified)
