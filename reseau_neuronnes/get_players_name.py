import json



def get_players_name(datas):
    """

    :return:
    """
    with open(datas, 'r') as file:
        json_data = json.load(file)
        players = []
        targets = []
        for person in json_data:
            if person["is_target"]:
                targets.append(person["nom"])
            if person["is_player"]:
                players.append(person["nom"])
        return players, targets
