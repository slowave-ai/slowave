"""Multilingual situation index for the standing-rule tier.

A *situation* is a coarse coding activity (commit, push/PR, dependencies, tests, ...).  Rules are
tagged with the situations they apply to once at write time, and an activation task is mapped to
situations at read time, so a rule the task text never names ("commit" vs "never add Co-Authored-By
lines") can still be recognised as applicable.

Two LLM-free signals feed the mapping:

* a typo-tolerant **lexicon** of stems in English, Italian, German, French, Spanish, Portuguese and
  Dutch (plus a few CJK terms), matched on accent-folded Unicode tokens, and
* **prototype sentences** per situation in the same languages, embedded once per process with the
  already-loaded multilingual encoder (the best language per situation wins).

Stems before the ``|`` marker are English and are also matched with typo tolerance.
Stem syntax: a stem matches a token by prefix; ``stem$`` demands the exact token; ``-stem`` matches
as a suffix of a long compound token ("integrationstest"); ``a_b$`` matches two adjacent tokens
("api_key$"); a stem starting with a CJK character is a substring of the text.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import get_close_matches

SPECS: dict[str, tuple[str, str]] = {
    "commit": ("commit", "commit amend squash cherry authored trailer conventional | コミット"),
    "push_pr": (
        "push/PR",
        "push pull$ pr$ prs$ merge rebas branch main$ fork upstream ramo$ unisci | rama$ "
        "ramas$ zweig pouss fusionn fusionar mergen pullrequest プッシュ プルリクエスト マージ ブランチ",
    ),
    "deps": (
        "dependencies",
        "install upgrad downgrad dependen npm$ pnpm yarn pip$ pipx poetry uv$ cargo lockfile "
        "lock$ bump package library libraries sdk$ node$ brew vendor dipendenz aggiorn "
        "pacchett libreri | abhangig bibliothek paket dependanc bibliotheque paquet "
        "bibliotec afhankelijk pakket 依存 パッケージ ライブラリ アップグレード",
    ),
    "tests": (
        "tests",
        "test spec$ specs$ pytest jest vitest playwright cypress e2e$ coverage flak assert "
        "mock fixture tdd$ regress snapshot collaud | regressie -test -tests テスト",
    ),
    "lint": (
        "lint/types",
        "lint eslint ruff mypy pyright typecheck tsc$ prettier format black$ isort typing "
        "type$ types$ formatt | import$ imports$ リント",
    ),
    "db_migration": (
        "migration",
        "migrat alembic flyway ddl$ column schema alter$ backfill index$ indexes migrazion "
        "colonn schem | indice$ indices$ migrac spalt kolom マイグレーション カラム スキーマ",
    ),
    "database": (
        "database",
        "sql queries database db$ postgres mysql sqlite table orm$ prisma transaction row "
        "rows delete$ select$ | query$ tabel tabla$ tablas$ requete datenbank abfrage "
        "consulta$ consultas$ donnees$ datos$ dados$ データベース テーブル",
    ),
    "deploy": (
        "deploy/release",
        "deploy release rollout roll$ rollback publish helm kubernetes k8s$ terraform canary "
        "hotfix vercel cd$ rilasci distribu pubblic | ship$ shipped$ staging desple desplie "
        "implant uitrol bereitstell デプロイ リリース ステージング",
    ),
    "logging": (
        "logging",
        "logger metric trace tracing telemetry monitor alert observab sentry structlog pii$ "
        "registr$ registri$ | log$ logs$ logg logfile registro$ registros$ journalis "
        "journaux$ protokoll ログ ロギング",
    ),
    "secrets": (
        "secrets/config",
        "secret token credential password passwd env$ envs$ environment$ apikey vault dotenv "
        "signing segret credenzial chiave$ chiavi$ | passwort geheimnis geheim schlussel "
        "kennwort umgebungsvariabl cle$ cles$ passe$ contrasena clave$ claves$ segred senha$ "
        "senhas$ chave$ chaves$ wachtwoord sleutel api_key$ api_keys$ private_key$ ssh_key$ "
        "パスワード シークレット 秘密 認証情報 apiキー",
    ),
    "api": (
        "API",
        "endpoint route routes api$ apis$ handler controller rest$ graphql http webhook "
        "pagin openapi respons idempoten richiest$ richieste$ | schnittstelle requisic "
        "エンドポイント",
    ),
    "money": (
        "money/dates",
        "money monetar amount currenc price pricing decimal payment invoice refund billing "
        "cents iban timestamp timezone tz$ utc$ datetime float round denaro importo importi "
        "valut fattur rimbors pagament | charge$ charges$ date$ dates$ centesim betrag "
        "rechnung zahlung erstattung preis wahrung geld montant remboursement facture "
        "paiement prix$ devise centime monnaie arrondi importe importes reembolso factura "
        "pago$ pagos$ precio moneda centavo divisa fecha$ fechas$ estorno fatura preco moeda "
        "data$ datas$ bedrag betaling factuur terugbetal prijs valuta datum zeitstempel "
        "zeitzone horodatage fuseau 金額 請求 返金 決済 通貨",
    ),
    "prod_ops": (
        "production",
        "prod$ production incident outage oncall runbook postmortem live$ customer stuck "
        "pager rollback hotfix produzion | producc producao plantao panne$ caida ausfall "
        "storung notfall productie storing down$ 本番 障害",
    ),
    "refactor": (
        "refactor",
        "refactor renam restructur rinomin riscriv cleanup tidy extract dedup simplif "
        "rewrite reorganiz move$ split$ | rifattor umbenenn umstruktur aufraum renomm "
        "renombr renome refator hernoem herstructur リファクタ",
    ),
    "review": (
        "review",
        "review audit approve feedback nit$ critique inspect | revision revisar revisa "
        "revisao revue$ relecture uberprufen beoordel nakijken レビュー",
    ),
    "docs": (
        "docs",
        "documentaz readme docstring changelog adr$ adrs$ wiki guide tutorial | doc$ docs$ "
        "document documentac dokument comments$ commento$ commenti$ comentario "
        "comentarios$ commentair kommentar ドキュメント",
    ),
    "frontend": (
        "frontend/UI",
        "component ui$ ux$ css button modal dialog react vue svelte style$ styles$ a11y "
        "accessib tailwind layout frontend jsx tsx icon aria html dom$ render i18n l10n "
        "translat locale | composant komponent bouton boton$ botones$ botao botoes knopf "
        "knop$ knoppen$ accesibl acessiv barrierefrei ボタン コンポーネント アクセシビリティ",
    ),
    "perf": (
        "performance",
        "slow perform latency optimi cache caching leak throughput profil benchmark "
        "bottleneck speed prestazion lentezz | lent$ lente$ lenteur lento$ lenta$ lentitud "
        "rendimiento desempenho langsam leistung traag snelheid prestatie eager 遅い パフォーマンス "
        "最適化 キャッシュ",
    ),
    "security": (
        "security/auth",
        "auth$ authentic authoriz authn$ authz$ login logout permission oauth jwt csrf xss "
        "injection vulnerab rbac encrypt sanitiz csp$ cors cve secure security cookie "
        "autenticaz autorizzaz sicurezz | authentif session$ sessions$ autentic autoriza "
        "connexion anmeld inloggen sicherheit securit seguridad seguranca veiligheid 認証 ログイン "
        "セキュリティ",
    ),
    "build_ci": (
        "build/CI",
        "build ci$ pipeline workflow docker compile webpack vite makefile actions github "
        "image container | compil ビルド パイプライン ワークフロー",
    ),
    "search": (
        "search",
        "find search grep locate where$ look$ rg$ fd$ usages callers references | cerca "
        "trova such cherch trouv busc procur zoek",
    ),
    "implement": (
        "implement",
        "add implement create write feature introduc develop support$ | aggiung crea$ creare "
        "ajout cree$ creer anad adicion crie$ crear erstell hinzu",
    ),
    "destructive": (
        "destructive ops",
        "delete remov rm$ drop wipe reset purge destroy truncat overwrite kill elimin "
        "cancell | force$ supprim effac loschen entfern borra exclu apag verwijder 削除 消去",
    ),
    "answer_style": (
        "answers",
        "explain summar answer reply respond report$ describe why$ done$ finish verify claim "
        "| spiega riassum erklar expliqu explica resum zusammenfass",
    ),
}
IDS: tuple[str, ...] = tuple(SPECS)
LABELS = {k: v[0] for k, v in SPECS.items()}

# One short sentence per language and situation (en, it, de, fr, es, pt).  The prototype of a
# situation is the best-matching of its sentences, so a task or rule in any of these languages
# lands near its situation in the multilingual embedding space.
PROTOTYPES: dict[str, tuple[str, ...]] = {
    "commit": (
        "Commit changes to git and write the commit message",
        "Fai il commit delle modifiche e scrivi il messaggio di commit",
        "Erstelle einen Git-Commit und schreibe die Commit-Nachricht",
        "Fais un commit des changements et écris le message de commit",
        "Haz un commit de los cambios y escribe el mensaje del commit",
        "Faça o commit das alterações e escreva a mensagem do commit",
    ),
    "push_pr": (
        "Push a branch and open a pull request for review",
        "Fai il push del ramo e apri una pull request per la revisione",
        "Pushe den Branch und eröffne einen Pull Request zur Prüfung",
        "Pousse la branche et ouvre une pull request pour relecture",
        "Haz push de la rama y abre un pull request para revisión",
        "Faça push da branch e abra um pull request para revisão",
    ),
    "deps": (
        "Install, upgrade or bump a package dependency and its lockfile",
        "Installa o aggiorna una dipendenza o libreria e il suo lockfile",
        "Installiere oder aktualisiere eine Paket-Abhängigkeit und das Lockfile",
        "Installe ou mets à jour une dépendance de paquet et son lockfile",
        "Instala o actualiza una dependencia del paquete y su lockfile",
        "Instale ou atualize uma dependência do pacote e o lockfile",
    ),
    "tests": (
        "Write or run automated tests and fix failing tests",
        "Scrivi o esegui test automatici e correggi i test che falliscono",
        "Schreibe oder starte automatisierte Tests und behebe fehlschlagende Tests",
        "Écris ou lance des tests automatisés et corrige les tests qui échouent",
        "Escribe o ejecuta tests automáticos y arregla los tests que fallan",
        "Escreva ou execute testes automatizados e corrija os testes que falham",
    ),
    "lint": (
        "Run the linter, formatter or type checker and fix style and type errors",
        "Esegui il linter, il formatter o il type checker e correggi gli errori di stile",
        "Führe Linter, Formatter oder Typprüfung aus und behebe Stil- und Typfehler",
        "Lance le linter, le formateur ou le vérificateur de types et corrige les erreurs",
        "Ejecuta el linter, el formateador o el verificador de tipos y corrige los errores",
        "Execute o linter, o formatador ou o verificador de tipos e corrija os erros",
    ),
    "db_migration": (
        "Write a database schema migration that adds or changes tables, columns or indexes",
        "Scrivi una migrazione dello schema del database che aggiunge colonne, tabelle o indici",
        "Schreibe eine Datenbank-Migration, die Spalten, Tabellen oder Indizes hinzufügt",
        "Écris une migration de schéma de base de données qui ajoute des colonnes ou des index",
        "Escribe una migración del esquema de la base de datos que añade columnas o índices",
        "Escreva uma migração do esquema do banco de dados que adiciona colunas ou índices",
    ),
    "database": (
        "Query or modify data in a SQL database",
        "Interroga o modifica i dati in un database SQL",
        "Frage Daten in einer SQL-Datenbank ab oder ändere sie",
        "Interroge ou modifie les données d'une base de données SQL",
        "Consulta o modifica datos en una base de datos SQL",
        "Consulte ou modifique dados em um banco de dados SQL",
    ),
    "deploy": (
        "Deploy or release a new version to staging or production",
        "Fai il deploy o rilascia una nuova versione in staging o in produzione",
        "Deploye oder veröffentliche eine neue Version auf Staging oder Produktion",
        "Déploie ou livre une nouvelle version en staging ou en production",
        "Despliega o publica una nueva versión en staging o en producción",
        "Faça o deploy ou lance uma nova versão em staging ou produção",
    ),
    "logging": (
        "Add logging, metrics, tracing or monitoring",
        "Aggiungi log, metriche, tracing o monitoraggio",
        "Füge Logging, Metriken, Tracing oder Monitoring hinzu",
        "Ajoute des logs, des métriques, du tracing ou de la supervision",
        "Añade logs, métricas, trazas o monitorización",
        "Adicione logs, métricas, tracing ou monitoramento",
    ),
    "secrets": (
        "Handle secrets, API keys, passwords and environment variables",
        "Gestisci segreti, chiavi API, password e variabili d'ambiente",
        "Verwalte Geheimnisse, API-Schlüssel, Passwörter und Umgebungsvariablen",
        "Gère les secrets, les clés d'API, les mots de passe et les variables d'environnement",
        "Gestiona secretos, claves de API, contraseñas y variables de entorno",
        "Gerencie segredos, chaves de API, senhas e variáveis de ambiente",
    ),
    "api": (
        "Design or implement an HTTP API endpoint or webhook",
        "Progetta o implementa un endpoint API HTTP o un webhook",
        "Entwerfe oder implementiere einen HTTP-API-Endpunkt oder Webhook",
        "Conçois ou implémente un endpoint d'API HTTP ou un webhook",
        "Diseña o implementa un endpoint de API HTTP o un webhook",
        "Projete ou implemente um endpoint de API HTTP ou um webhook",
    ),
    "money": (
        "Handle money amounts, currencies, prices, dates, time zones and number formats",
        "Gestisci importi di denaro, valute, prezzi, date e fusi orari",
        "Behandle Geldbeträge, Währungen, Preise, Daten und Zeitzonen",
        "Gère les montants d'argent, les devises, les prix, les dates et les fuseaux horaires",
        "Gestiona importes de dinero, monedas, precios, fechas y zonas horarias",
        "Trate valores monetários, moedas, preços, datas e fusos horários",
    ),
    "prod_ops": (
        "Operate on production systems, handle an incident or outage",
        "Opera sui sistemi di produzione e gestisci un incidente o un'interruzione",
        "Arbeite an Produktivsystemen und behandle einen Vorfall oder Ausfall",
        "Interviens sur les systèmes de production et gère un incident ou une panne",
        "Opera sobre sistemas de producción y gestiona un incidente o una caída",
        "Opere sistemas de produção e trate um incidente ou uma queda",
    ),
    "refactor": (
        "Refactor, rename or restructure existing code without changing behavior",
        "Rifattorizza, rinomina o ristruttura il codice esistente senza cambiarne il comportamento",
        "Refaktoriere, benenne um oder strukturiere bestehenden Code ohne Verhaltensänderung",
        "Refactorise, renomme ou restructure le code existant sans changer le comportement",
        "Refactoriza, renombra o reestructura el código existente sin cambiar el comportamiento",
        "Refatore, renomeie ou reestruture o código existente sem mudar o comportamento",
    ),
    "review": (
        "Review a pull request or audit code changes",
        "Rivedi una pull request o controlla le modifiche al codice",
        "Prüfe einen Pull Request oder begutachte Codeänderungen",
        "Relis une pull request ou audite les changements de code",
        "Revisa un pull request o audita los cambios de código",
        "Revise um pull request ou audite as mudanças de código",
    ),
    "docs": (
        "Write or update documentation, README, comments or design documents",
        "Scrivi o aggiorna la documentazione, il README, i commenti o i documenti di design",
        "Schreibe oder aktualisiere Dokumentation, README, Kommentare oder Designdokumente",
        "Écris ou mets à jour la documentation, le README, les commentaires ou les documents",
        "Escribe o actualiza la documentación, el README, los comentarios o los documentos",
        "Escreva ou atualize a documentação, o README, os comentários ou os documentos",
    ),
    "frontend": (
        "Build a user interface component with accessibility, styling and translations",
        "Costruisci un componente di interfaccia utente con accessibilità, stile e traduzioni",
        "Baue eine Benutzeroberflächen-Komponente mit Barrierefreiheit, Styling und Übersetzungen",
        "Construis un composant d'interface utilisateur avec accessibilité, style et traductions",
        "Construye un componente de interfaz de usuario con accesibilidad, estilos y traducciones",
        "Construa um componente de interface do usuário com acessibilidade, estilo e traduções",
    ),
    "perf": (
        "Investigate slow performance, latency, caching or memory usage",
        "Indaga su prestazioni lente, latenza, cache o uso di memoria",
        "Untersuche langsame Leistung, Latenz, Caching oder Speicherverbrauch",
        "Enquête sur des performances lentes, la latence, le cache ou l'usage mémoire",
        "Investiga el bajo rendimiento, la latencia, la caché o el uso de memoria",
        "Investigue desempenho lento, latência, cache ou uso de memória",
    ),
    "security": (
        "Implement authentication, authorization or fix a security vulnerability",
        "Implementa autenticazione, autorizzazione o correggi una vulnerabilità di sicurezza",
        "Implementiere Authentifizierung, Autorisierung oder behebe eine Sicherheitslücke",
        "Implémente l'authentification, l'autorisation ou corrige une faille de sécurité",
        "Implementa autenticación, autorización o corrige una vulnerabilidad de seguridad",
        "Implemente autenticação, autorização ou corrija uma vulnerabilidade de segurança",
    ),
    "build_ci": (
        "Change the build, Docker image, CI pipeline or compile configuration",
        "Modifica la build, l'immagine Docker, la pipeline CI o la configurazione di compilazione",
        "Ändere den Build, das Docker-Image, die CI-Pipeline oder die Kompilierungskonfiguration",
        "Modifie le build, l'image Docker, le pipeline CI ou la configuration de compilation",
        "Cambia el build, la imagen Docker, el pipeline de CI o la configuración de compilación",
        "Altere o build, a imagem Docker, o pipeline de CI ou a configuração de compilação",
    ),
    "search": (
        "Search the codebase to find where something is defined or used",
        "Cerca nel codice dove qualcosa è definito o usato",
        "Durchsuche den Code, um zu finden, wo etwas definiert oder verwendet wird",
        "Cherche dans le code où quelque chose est défini ou utilisé",
        "Busca en el código dónde se define o se usa algo",
        "Procure no código onde algo é definido ou usado",
    ),
    "implement": (
        "Implement a new feature by writing new code",
        "Implementa una nuova funzionalità scrivendo nuovo codice",
        "Implementiere eine neue Funktion, indem du neuen Code schreibst",
        "Implémente une nouvelle fonctionnalité en écrivant du nouveau code",
        "Implementa una nueva funcionalidad escribiendo código nuevo",
        "Implemente uma nova funcionalidade escrevendo código novo",
    ),
    "destructive": (
        "Run destructive commands such as deleting files, resetting git or dropping data",
        "Esegui comandi distruttivi come cancellare file, resettare git o eliminare dati",
        "Führe destruktive Befehle aus, etwa Dateien löschen, Git zurücksetzen oder Daten verwerfen",
        "Exécute des commandes destructrices comme supprimer des fichiers ou réinitialiser git",
        "Ejecuta comandos destructivos como borrar archivos, resetear git o eliminar datos",
        "Execute comandos destrutivos como apagar arquivos, resetar o git ou excluir dados",
    ),
    "answer_style": (
        "Explain, summarize or report the result to the user in the right language and tone",
        "Spiega, riassumi o riferisci il risultato all'utente nella lingua e nel tono giusti",
        "Erkläre, fasse zusammen oder berichte dem Nutzer das Ergebnis in passender Sprache",
        "Explique, résume ou rapporte le résultat à l'utilisateur dans la bonne langue",
        "Explica, resume o informa el resultado al usuario en el idioma y tono adecuados",
        "Explique, resuma ou relate o resultado ao usuário no idioma e tom adequados",
    ),
}
PROTOTYPE_LANGS = 6

# Tokens that merely start like a stem ("testo" is Italian for "text", not a test).
_NOT_TOKENS = frozenset({"testo", "testi", "testa", "testigo", "testimonio", "testament"})
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
# Product / library names that contain a situation word but do not mean the situation.
_NAMES_RE = re.compile(r"\b(tanstack|react|vue|apollo|urql)[ -]query\b|\bswr\b")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_SUFFIX_MIN_EXTRA = 4


def normalize(text: str) -> str:
    """Casefold and strip accents (``ausgeführt`` -> ``ausgefuhrt``), keep non-Latin scripts."""
    folded = unicodedata.normalize("NFKD", str(text or "").casefold())
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize(text))


def _is_cjk(ch: str) -> bool:
    return ord(ch) >= 0x2E80


def _build() -> tuple[
    dict[str, list[str]],
    dict[int, dict[str, list[str]]],
    dict[str, list[str]],
    list[tuple[str, str]],
    list[str],
]:
    exact: dict[str, list[str]] = {}
    prefix: dict[int, dict[str, list[str]]] = {}
    suffix: dict[str, list[str]] = {}
    cjk: list[tuple[str, str]] = []
    vocab: list[str] = []
    for sid, (_, stems) in SPECS.items():
        fuzzy_ok = True  # stems before the "|" marker are English: only they are typo-matched
        for stem in stems.split():
            if stem == "|":
                fuzzy_ok = False
                continue
            if _is_cjk(stem[0]) or (not stem.isascii() and _is_cjk(stem[-1])):
                cjk.append((normalize(stem), sid))
            elif stem.endswith("$"):
                exact.setdefault(stem[:-1], []).append(sid)
            elif stem.startswith("-"):
                suffix.setdefault(stem[1:], []).append(sid)
            else:
                prefix.setdefault(len(stem), {}).setdefault(stem, []).append(sid)
                if fuzzy_ok and len(stem) >= 5:
                    vocab.append(stem)
    return exact, prefix, suffix, cjk, sorted(set(vocab))


_EXACT, _PREFIX, _SUFFIX, _CJK, _VOCAB = _build()
_PREFIX_LENS = sorted(_PREFIX)
_VOCAB_BY_FIRST: dict[str, list[str]] = {}
for _v in _VOCAB:
    _VOCAB_BY_FIRST.setdefault(_v[0], []).append(_v)


def lexicon_situations(text: str, *, fuzzy: bool = False) -> dict[str, float]:
    """Map text to situations by lexicon.  ``fuzzy`` adds typo tolerance ('upgrde' -> 'upgrad')."""
    value = _NAMES_RE.sub(" ", _URL_RE.sub(" ", normalize(text)))
    found: dict[str, float] = {}
    toks = [t for t in _TOKEN_RE.findall(value) if t not in _NOT_TOKENS]
    for a, b in zip(toks, toks[1:]):
        for sid in _EXACT.get(f"{a}_{b}", ()):
            found[sid] = 1.0
    for tok in toks:
        hit = False
        for sid in _EXACT.get(tok, ()):
            found[sid] = 1.0
            hit = True
        for ln in _PREFIX_LENS:
            if len(tok) < ln:
                break
            for sid in _PREFIX[ln].get(tok[:ln], ()):
                found[sid] = 1.0
                hit = True
        for suf, sids in _SUFFIX.items():
            if len(tok) >= len(suf) + _SUFFIX_MIN_EXTRA and tok.endswith(suf):
                for sid in sids:
                    found[sid] = 1.0
                hit = True
        if fuzzy and not hit and len(tok) >= 5 and tok.isalpha() and tok.isascii():
            cands = [v for v in _VOCAB_BY_FIRST.get(tok[0], ()) if abs(len(v) - len(tok)) <= 2]
            for match in get_close_matches(tok, cands, n=1, cutoff=0.8):
                for sid in _PREFIX[len(match)][match]:
                    found.setdefault(sid, 0.9)
    if not value.isascii():
        for stem, sid in _CJK:
            if stem in value:
                found[sid] = 1.0
    return found
