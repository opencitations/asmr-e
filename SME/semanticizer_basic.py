import json
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD

MITO = Namespace("http://purl.org/spar/mito#")
FABIO = Namespace("http://purl.org/spar/fabio#")
DCTERMS = Namespace("http://purl.org/dc/terms/")
PRISM = Namespace("http://prismstandard.org/namespaces/basic/2.0/")
PROV = Namespace("http://www.w3.org/ns/prov#") 
EX = Namespace("http://example.org/") 

g = Graph()
g.bind("mito", MITO)
g.bind("fabio", FABIO)
g.bind("dcterms", DCTERMS)
g.bind("prism", PRISM)
g.bind("prov", PROV)
g.bind("ex", EX) 
input_file = r"C:\Users\fabio\Desktop\Matteo\DHDK\Progetto_tesi\SME_JSON\results\SoftCite\gemma\9b\fp16\RAG\sentence\converted_results_test_4.jsonl"
with open(input_file, "r") as f:
    for line in f:
        record = json.loads(line.strip())
        doc_id = record["id"]
        
        journal_article = URIRef(EX[f"JournalArticle-{doc_id}"])
        g.add((journal_article, RDF.type, FABIO.JournalArticle)) 
        g.add((journal_article, RDF.type, MITO.MentioningEntity))  
        g.add((journal_article, DCTERMS.identifier, Literal(doc_id))) 
        
        for software in record["software"]:
            software_name = software["name"]
            
            software_uri = URIRef(EX[f"software/{software_name.replace(' ', '')}"])
            
            g.add((software_uri, RDF.type, MITO.MentionedEntity))
            g.add((software_uri, DCTERMS.title, Literal(software_name)))  
            
            mention_id = f"mention-{doc_id}-{software_name.replace(' ', '_')}"
            mention = URIRef(EX[mention_id])
            g.add((mention, RDF.type, MITO.Mention))
            g.add((mention, MITO.hasMentioningEntity, journal_article))
            g.add((mention, MITO.hasMentionedEntity, software_uri))
            g.add((mention, MITO.hasMentionType, URIRef("http://purl.org/spar/mito#explicit-mention"))) 
                        
            g.add((journal_article, MITO.mentions, software_uri))
            g.add((software_uri, MITO.isMentionedBy, journal_article))
            
            if software.get("url"):
                for url in software["url"]:
                    g.add((software_uri, FABIO.hasURL, Literal(url.strip())))
            if software.get("version"):
                for version in software["version"]:
                    g.add((software_uri, PRISM.versionIdentifier, Literal(version)))
            if software.get("language"):
                for language in software["language"]:
                    g.add((software_uri, DCTERMS.language, Literal(language)))
            if software.get("publisher"):
                for publisher in software["publisher"]:
                    g.add((software_uri, DCTERMS.publisher, Literal(publisher)))

output_file = r"C:\Users\fabio\Desktop\Matteo\DHDK\Progetto_tesi\asmr-e\data\processed\output_basic_generic_prefix4.rdf"
g.serialize(destination=output_file, format="xml")
output_file_ttl = r"C:\Users\fabio\Desktop\Matteo\DHDK\Progetto_tesi\asmr-e\data\processed\output_basic_generic_prefix4.ttl"
g.serialize(destination=output_file_ttl, format="turtle")
print(f"RDF saved in RDF/XML format: {output_file}")
print(f"RDF saved in Turtle format: {output_file_ttl}")