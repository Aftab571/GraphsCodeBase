from neo4j import GraphDatabase
import pandas as pd

class Connection:

    def fetch_data(self,query, params={}):
        with self.driver.session() as session:
            result = session.run(query, params)
            #return result
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

    def __init__(self,data):
        self.driver = GraphDatabase.driver(data["server_uri"], auth=(data["admin_user"], data["admin_pass"]))
        


