# Authentication Types and Identity Protocols

A structured reference covering credential schemes, session and token-based authentication, JWT internals, OAuth 2.0, OpenID Connect, SSO/federation, SAML, and Keycloak as a reference IdP implementation.

---

## Table of Contents

1. [Authentication vs. Authorization](#1-authentication-vs-authorization)
2. [Authentication Factors](#2-authentication-factors)
3. [Evolution of Authentication](#3-evolution-of-authentication)
4. [Credential-Based Schemes](#4-credential-based-schemes)
   - 4.1 Basic Authentication
   - 4.2 Digest Authentication
   - 4.3 API Key Authentication
5. [Session-Based Authentication](#5-session-based-authentication)
6. [Token-Based Authentication](#6-token-based-authentication)
   - 6.1 Bearer Tokens
   - 6.2 Opaque vs. Self-Contained Tokens
7. [JSON Web Tokens (JWT)](#7-json-web-tokens-jwt)
8. [Access Tokens and Refresh Tokens](#8-access-tokens-and-refresh-tokens)
9. [The Bearer Security Model: Possession, Theft, and Proof-of-Possession](#9-the-bearer-security-model)
10. [OAuth 2.0](#10-oauth-20)
11. [OpenID Connect (OIDC)](#11-openid-connect-oidc)
12. [Single Sign-On and Identity Federation](#12-single-sign-on-and-identity-federation)
13. [SAML 2.0](#13-saml-20)
14. [Keycloak: A Reference IdP Implementation](#14-keycloak-a-reference-idp-implementation)
15. [Comparison Tables](#15-comparison-tables)
16. [Security Notes and Common Pitfalls](#16-security-notes-and-common-pitfalls)

---

## 1. Authentication vs. Authorization

**Authentication** answers *"who are you?"*. **Authorization** answers *"what are you allowed to do?"*. The two are frequently bundled into a single login flow but are conceptually and, in modern protocols, mechanically distinct (this is precisely the split OAuth 2.0 vs. OIDC formalizes — see §10–11).

| Concept | Question | Example |
|---|---|---|
| Authentication | Who are you? | "This is Mehrab." |
| Authorization | What can you do? | "Mehrab can push to this repository." |

---

## 2. Authentication Factors

Authentication mechanisms rely on one or more of the following factor classes. Using two or more *independent* classes together is what defines multi-factor authentication (MFA) — combining two secrets from the *same* class (e.g., a password and a PIN) is not MFA.

### 2.1 Something You Know (knowledge factor)
Passwords, PINs, security questions. Weakness: fully replicable by anyone who learns the secret; no binding to a physical entity.

### 2.2 Something You Have (possession factor)
Phones (OTP/push), hardware security keys (FIDO2/U2F), smart cards, TOTP tokens. Weakness: device loss/theft, SIM-swap attacks against SMS OTP specifically (SMS OTP is the weakest possession factor for this reason).

### 2.3 Something You Are (inherence factor)
Fingerprint, face, iris. Note: biometrics identify but, unlike passwords, cannot be rotated if compromised — a leaked fingerprint template is permanent. Systems should store salted templates or use secure enclaves (e.g., Secure Enclave/TEE), never raw biometric data.

### 2.4 Somewhere You Are (contextual factor)
IP range, GPS, network origin (e.g., corporate VPN). This is a risk-signal, not a standalone authentication method — it's normally combined with another factor for step-up or conditional access decisions.

```mermaid
flowchart LR
    User --> Password["Something you know"]
    User --> Phone["Something you have"]
    User --> Bio["Something you are"]
    Password --> MFA
    Phone --> MFA
    Bio --> MFA
    MFA --> Access
```

---

## 3. Evolution of Authentication

```mermaid
timeline
    title Evolution of Authentication
    1990 : Username + Password
    1993 : HTTP Basic Authentication
    1997 : HTTP Digest Authentication
    2000 : Server-side Session Authentication
    2005 : API Keys for web APIs
    2010 : OAuth 1.0a / early Bearer Tokens
    2012 : OAuth 2.0 (RFC 6749)
    2014 : OpenID Connect
    2015 : JWT (RFC 7519) becomes mainstream
    2019 : WebAuthn / FIDO2 passkeys
```

Note on dates: RFC 6749 (OAuth 2.0) and RFC 7519 (JWT) were published in 2012 and 2015 respectively; JWT is often used as the *token format* inside OAuth/OIDC responses rather than a competing standard, so "JWT" and "OAuth 2.0" are not mutually exclusive rungs on this timeline — they are frequently layered.

---

## 4. Credential-Based Schemes

### 4.1 Basic Authentication

The client sends `username:password` in the `Authorization` header, Base64-encoded.

> **Base64 is encoding, not encryption.** It is trivially reversible and provides zero confidentiality on its own — Basic Auth is only acceptable over TLS.

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    C->>S: GET /private-resource
    S-->>C: 401 Unauthorized (WWW-Authenticate: Basic)
    C->>C: Base64(username:password)
    C->>S: Authorization: Basic <base64>
    S->>S: Decode and verify credentials
    S-->>C: 200 OK
```

```http
GET /profile HTTP/1.1
Authorization: Basic bWVocmFiOnBhc3N3b3JkMTIz
```

**Advantages:** trivial to implement, no server-side session state.
**Disadvantages:** the password is retransmitted on *every* request; no expiration; no scoping; must run over TLS or credentials are exposed in plaintext on the wire.
**Use for:** internal tools, local testing, service-to-service calls behind a private network. **Avoid for:** anything public-facing.

### 4.2 Digest Authentication

Digest Auth avoids sending the password itself. The server issues a `nonce` (a server-generated, single-use challenge value); the client returns a hash rather than the raw credential.

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    C->>S: Request protected resource
    S-->>C: 401 (WWW-Authenticate: Digest nonce="abc123")
    C->>C: response = H(H(username:realm:password):nonce:H(method:URI))
    C->>S: Authorization: Digest response=...
    S->>S: Recompute expected response, compare
    S-->>C: 200 OK
```

Technical correction to the common simplified explanation: RFC 7616 digest response is not simply `hash(username, password, nonce)`. It's computed from two intermediate hashes — **HA1** = `H(username:realm:password)` and **HA2** = `H(method:digestURI)` — combined with the nonce (and, when `qop` is used, a client nonce `cnonce` and nonce count `nc`) as `H(HA1:nonce:nc:cnonce:qop:HA2)`. The `realm` and `qop` (quality of protection) parameters matter for correctness and replay resistance.

**Advantages:** password not sent in the clear even without TLS.
**Disadvantages:** legacy MD5 usage in most implementations is cryptographically weak; complex to implement correctly; largely superseded by TLS + Basic/Bearer or token-based schemes. Rarely used in new systems today.

### 4.3 API Key Authentication

An opaque, application-scoped secret sent via a header (commonly `X-API-Key` or `Authorization: ApiKey <key>`).

```mermaid
sequenceDiagram
    participant App as Application
    participant API as API Server
    participant DB as Key Store
    App->>API: GET /data (X-API-Key: abc123)
    API->>DB: Look up key
    DB-->>API: Key metadata + scopes
    API->>API: Check permissions
    API-->>App: Response
```

**Key property:** an API key typically identifies an *application or integration*, not an individual human user — it carries no user-level authorization context unless the backend explicitly maps it to one.

| | Password | API Key |
|---|---|---|
| Represents | A user | An application/integration |
| Rotation | Manual, disruptive | Straightforward to rotate |
| Expiration | Often none by default | Often none by default (should be added) |
| Typical exposure risk | Phishing, reuse | Accidental commit to source control |

---

## 5. Session-Based Authentication

The server holds authentication state; the client holds only an opaque reference to it (a session ID, typically delivered via a cookie).

```mermaid
sequenceDiagram
    participant B as Browser
    participant S as Server
    participant DB as User DB
    participant SS as Session Store
    B->>S: POST /login (username, password)
    S->>DB: Verify credentials
    DB-->>S: Valid
    S->>SS: Create session {user_id, role, expiry}
    SS-->>S: session_id = abc987
    S-->>B: Set-Cookie: SESSIONID=abc987
    Note over B,S: Subsequent request
    B->>S: GET /profile (Cookie: SESSIONID=abc987)
    S->>SS: Lookup abc987
    SS-->>S: user_id=123
    S-->>B: Profile data
```

**Cookie hardening** — a production session cookie should set:

```http
Set-Cookie: SESSIONID=abc987; HttpOnly; Secure; SameSite=Strict
```

- `HttpOnly` — blocks JavaScript access, mitigating token exfiltration via XSS.
- `Secure` — cookie is only sent over HTTPS.
- `SameSite=Strict` (or `Lax`, depending on cross-site navigation needs) — mitigates CSRF by restricting cross-origin cookie transmission.

**Advantages:** instant, server-enforced revocation (delete the session record); no client-side token to secure long-term.
**Disadvantages:** server must maintain state, which complicates horizontal scaling — a session created on Server A isn't visible to Server B unless sessions live in a shared store (e.g., Redis) or sticky routing is used.

---

## 6. Token-Based Authentication

Token authentication shifts the credential from a server-held session to a client-held artifact, sent with every request.

```mermaid
sequenceDiagram
    participant C as Client
    participant AS as Auth Server
    participant API as Resource Server
    C->>AS: Login (username/password)
    AS-->>C: Access Token
    C->>API: GET /resource (Authorization: Bearer <token>)
    API->>API: Validate token (locally or via introspection)
    API-->>C: Protected data
```

### 6.1 Bearer Tokens

"Bearer" describes *how the token is used*: whoever presents it is granted the access it represents, with no additional proof required. This is analogous to physical cash or a hotel key card — the system verifies the artifact is valid, not that the presenter is its original, rightful owner.

```http
GET /api/users HTTP/1.1
Authorization: Bearer eyJhbGciOiJIUzI1...
```

**Consequence:** bearer token theft is equivalent to account compromise for the token's lifetime and scope. This holds regardless of the token's internal format — it applies equally to opaque tokens and signed JWTs (elaborated in §9).

### 6.2 Opaque vs. Self-Contained Tokens

| | Opaque token | Self-contained token (JWT) |
|---|---|---|
| Format | Random string, no embedded meaning | `header.payload.signature`, embedded claims |
| Validation | Requires a server-side lookup (DB or introspection call) | Can be validated locally via signature check — no round trip needed |
| Revocation | Immediate (delete the DB row) | Hard — the token remains cryptographically valid until it expires, unless a separate revocation/blacklist mechanism is added |
| Best fit | Systems needing instant revocation, small-to-medium scale | Distributed systems, microservices, where avoiding a shared lookup on every request matters |

---

## 7. JSON Web Tokens (JWT)

A JWT (RFC 7519) has three Base64URL-encoded segments: `HEADER.PAYLOAD.SIGNATURE`.

```mermaid
flowchart LR
    JWT --> Header
    JWT --> Payload
    JWT --> Signature
```

**Header** — algorithm and token type:
```json
{ "alg": "HS256", "typ": "JWT" }
```

**Payload** — claims. Standard/registered claims include `sub` (subject), `iss` (issuer), `aud` (audience), `exp` (expiration), `iat` (issued at), `nbf` (not before); anything else is a custom claim:
```json
{ "sub": "123456", "name": "Mehrab", "role": "admin", "exp": 1789981123 }
```

> **The payload is Base64URL-encoded, not encrypted.** Anyone holding the token can decode and read it. Never place secrets, passwords, or sensitive PII directly in a JWT payload — use JWE (JSON Web Encryption) if confidentiality of the claims themselves is required, which is a separate mechanism from the signed JWT (JWS) discussed here.

**Signature** — for the common symmetric case:
```
signature = HMAC-SHA256(
    base64url(header) + "." + base64url(payload),
    secret_key
)
```
For asymmetric algorithms (RS256, ES256), the issuer signs with a private key and any verifier checks with the corresponding public key — this is what allows *stateless, distributed* verification without every service sharing a secret.

### Verification steps (a correct verifier must do all of these — a common bug is checking only the signature)

```mermaid
sequenceDiagram
    participant C as Client
    participant API as Server
    C->>API: Authorization: Bearer <JWT>
    API->>API: 1. Verify signature against expected alg + key
    API->>API: 2. Reject if alg is unexpected (esp. "none")
    API->>API: 3. Check exp / nbf
    API->>API: 4. Check iss and aud match expected values
    API-->>C: Access granted or 401
```

### Known JWT pitfalls

- **Algorithm confusion / "none" attack** — a naive verifier that trusts the `alg` field from the token itself can be tricked into accepting an unsigned token (`alg: none`) or into verifying an RS256 token's signature using the public key as if it were an HMAC secret. Verifiers must pin the expected algorithm server-side rather than trusting the token header.
- **No native revocation** — a JWT remains valid until `exp`, even after logout or privilege change. Common mitigations: short expirations (minutes), a server-side revocation/deny-list checked for security-sensitive operations, or refresh-token rotation (§8).
- **Payload is public** — never embed secrets or unnecessary PII.

**Advantages:** stateless verification, horizontally scalable without a shared session store.
**Disadvantages:** the revocation problem above; larger than opaque tokens (extra transmitted bytes on every request).

---

## 8. Access Tokens and Refresh Tokens

Modern systems split tokens by purpose and lifetime to bound the blast radius of theft.

| | Access Token | Refresh Token |
|---|---|---|
| Purpose | Authorizes API calls directly | Used only to obtain a new access token |
| Typical lifetime | Minutes | Days to months |
| Sent to resource APIs | Yes | No — sent only to the authorization server's token endpoint |
| Recommended storage | Memory (avoid `localStorage`) | Secure, `HttpOnly` storage; rotate on use |

```mermaid
flowchart LR
    Login --> AccessToken[Access Token]
    Login --> RefreshToken[Refresh Token]
    AccessToken --> APICall[API Request]
    APICall --> Expired{Expired?}
    Expired -->|Yes| RefreshRequest[POST /token with refresh_token]
    RefreshToken --> RefreshRequest
    RefreshRequest --> AuthServer[Authorization Server]
    AuthServer --> NewAccessToken[New Access Token]
```

**Refresh token rotation:** best practice issues a *new* refresh token on every use and invalidates the old one. If a previously-invalidated refresh token is presented again, this signals theft — the authorization server should revoke the entire token family, not just reject the one call.

---

## 9. The Bearer Security Model

This section consolidates the possession/theft/PoP discussion, since it applies uniformly across cookies, opaque bearer tokens, JWTs, OAuth access tokens, and SAML/OIDC identity artifacts.

**Core principle:** a digital signature (as in a JWT or SAML assertion) proves two things —

1. The artifact was issued by a trusted party.
2. The artifact has not been modified since issuance.

It does **not** prove that the current presenter is the artifact's original, legitimate owner. Signature validity and rightful possession are independent properties.

```mermaid
flowchart TB
    Artifact["Signed Artifact (JWT / SAML Assertion)"]
    Artifact --> Sig["Signature Valid?"]
    Sig -->|Yes| Trust["Issued by trusted party, unmodified"]
    Sig -->|No proof of| Owner["Current holder is the rightful owner"]
```

If an artifact is stolen — via XSS, a compromised network, a leaked log, or a malicious browser extension — the attacker can present it exactly as the legitimate user would, and the receiving system generally cannot distinguish the two, until the artifact expires, is revoked, or additional binding is enforced.

**Mitigations in widespread use:**
- Short-lived access tokens/assertions (minutes) to shrink the exploitation window.
- Refresh token rotation with theft detection (§8).
- Transport security (TLS) to prevent interception in transit.
- Cookie hardening (`HttpOnly`, `Secure`, `SameSite`) for session cookies specifically.
- **Sender-constrained (Proof-of-Possession) tokens** — cryptographically bind the token to the client, so possession of the token alone is insufficient:
  - **DPoP** (Demonstrating Proof-of-Possession, RFC 9449) — the client signs each request with a private key; the server checks the signature against a public key bound into the token.
  - **Mutual TLS (mTLS)** — the client authenticates via a TLS client certificate, and the token is bound to that certificate.

```mermaid
flowchart LR
    subgraph Bearer["Bearer model"]
        T1[Token] --> A1[Access granted]
    end
    subgraph PoP["Proof-of-Possession model"]
        T2[Token] --> K[+ Private key / mTLS cert]
        K --> A2[Access granted]
    end
```


---

## 10. OAuth 2.0

OAuth 2.0 (RFC 6749) is an **authorization** framework. It answers *"what can this application access on the user's behalf?"* — not *"who is the user?"* (that's OIDC's job, §11). It exists so a user can grant a third-party application scoped access to their resources without ever handing over their password.

### Roles

- **Resource Owner** — the user (e.g., Mehrab).
- **Client** — the application requesting access.
- **Authorization Server** — authenticates the user and issues tokens (e.g., a Keycloak or Google identity platform instance).
- **Resource Server** — the API hosting the protected resource.

```mermaid
flowchart LR
    User --> Client
    Client --> AuthorizationServer[Authorization Server]
    AuthorizationServer --> AccessToken[Access Token]
    AccessToken --> ResourceServer[Resource Server]
    ResourceServer --> ProtectedAPI[Protected API]
```

### Authorization Code Flow (the standard, recommended flow)

```mermaid
sequenceDiagram
    participant U as User
    participant C as Client App
    participant AS as Authorization Server
    participant RS as Resource Server
    U->>C: Initiate login
    C->>AS: Redirect: authorization request
    AS->>U: Login + consent screen
    U->>AS: Authenticate, approve scopes
    AS-->>C: Authorization code (via redirect)
    C->>AS: Exchange code for tokens (client_id, client_secret, code)
    AS-->>C: Access token (+ refresh token)
    C->>RS: API request (Bearer access token)
    RS-->>C: Protected resource
```

The authorization code itself is short-lived and single-use — it is *not* the access token; it's a temporary voucher exchanged for the actual token in a back-channel (server-to-server) call, which keeps the token off the browser's redirect URL and browser history.

### Grant Types

| Grant | Use case | Status |
|---|---|---|
| Authorization Code (+ PKCE) | Web apps, mobile apps, SPAs | Recommended standard for anything with a user |
| Client Credentials | Machine-to-machine, no user involved | Standard for service-to-service |
| Device Authorization | Browserless devices (Smart TVs, CLIs) | Standard for that use case |
| Implicit | Browser apps returning the token directly in the redirect fragment | **Deprecated** — superseded by Authorization Code + PKCE, which avoids exposing tokens in browser history/logs |
| Resource Owner Password Credentials (ROPC) | App collects username/password directly and exchanges for a token | **Discouraged** — reintroduces the exact problem OAuth exists to avoid (the app sees the password); acceptable only for highly trusted first-party legacy migrations |

**Client Credentials flow (machine-to-machine):**
```mermaid
sequenceDiagram
    participant A as Service A
    participant AS as Authorization Server
    participant B as Service B API
    A->>AS: client_id + client_secret
    AS-->>A: Access token
    A->>B: Bearer token
    B-->>A: Response
```

### PKCE (Proof Key for Code Exchange, RFC 7636)

Public clients (mobile apps, SPAs) cannot safely embed a `client_secret` — it can be extracted from the app binary or bundle. PKCE closes this gap without requiring a secret:

1. Client generates a random `code_verifier`.
2. Client derives `code_challenge = BASE64URL(SHA256(code_verifier))` and sends the challenge (not the verifier) with the initial authorization request.
3. At the token exchange step, the client sends the original `code_verifier`.
4. The authorization server recomputes the hash and confirms it matches the challenge it stored — proving the entity exchanging the code is the same one that started the flow.

```mermaid
sequenceDiagram
    participant C as Client
    participant AS as Authorization Server
    C->>C: Generate code_verifier, derive code_challenge = SHA256(verifier)
    C->>AS: Authorization request + code_challenge
    AS-->>C: Authorization code
    C->>AS: Token request: code + code_verifier
    AS->>AS: Verify SHA256(verifier) == stored challenge
    AS-->>C: Access token
```

PKCE defends specifically against **authorization code interception** — e.g., a malicious app registered to intercept the same custom URI scheme on a mobile OS. As of OAuth 2.1 (the consolidated best-practices successor draft), PKCE is required for *all* authorization code flows, not just public clients, as defense in depth.

### Scopes and Token Introspection

Scopes (`scope=email profile photos.read`) express the specific permissions granted — the client can read but, per this example, not delete photos.

For opaque access tokens, a resource server can't inspect the token itself, so it calls the authorization server's **introspection endpoint** (RFC 7662) to check validity and retrieve associated scopes/claims in real time. This is the opaque-token equivalent of a JWT's local signature check, at the cost of a network round trip — but with the benefit of instant revocation visibility.

---

## 11. OpenID Connect (OIDC)

OIDC is an identity layer built directly on top of OAuth 2.0. Where OAuth answers "what can this app access," OIDC answers "who is the user," by adding a new artifact — the **ID Token** — and a reserved `openid` scope.

| | OAuth 2.0 | OIDC |
|---|---|---|
| Purpose | Authorization | Authentication |
| Core question | What can the app access? | Who is the user? |
| Artifact | Access Token (any format) | ID Token (JWT, standardized claims) |

```mermaid
sequenceDiagram
    participant U as User
    participant C as Client
    participant OP as OpenID Provider
    U->>C: Login
    C->>OP: Authorization request (scope=openid profile email)
    OP->>U: Authenticate
    OP-->>C: Authorization code
    C->>OP: Exchange code
    OP-->>C: ID Token + Access Token
    C->>C: Validate ID Token (sig, iss, aud, exp)
    C->>U: Session established
```

**ID Token payload example:**
```json
{
  "iss": "https://identity-provider.com",
  "sub": "987654",
  "aud": "my_application",
  "exp": 1789981123,
  "email": "user@example.com"
}
```

**Important distinction the ID Token is often misused for:** the ID Token is meant to tell *the client application* who just authenticated — it is not designed to be sent to resource APIs as an authorization credential. Using it as a bearer token for API calls (instead of the Access Token) is a common anti-pattern, since the ID Token's `aud` claim is scoped to the client, not to arbitrary resource servers.

---

## 12. Single Sign-On and Identity Federation

SSO lets a user authenticate once with a central **Identity Provider (IdP)** and gain access to multiple independent applications (**Service Providers**, SPs) without re-entering credentials for each.

### The two layers of an SSO flow

It helps to separate these explicitly, since they're often conflated:

1. **User ↔ IdP authentication** — how the user proves their identity to the IdP. This can be *any* mechanism: password, MFA, certificate, smart card, biometric, passwordless. The applications relying on SSO never need to know which method was used.
2. **IdP ↔ Service Provider trust** — how the SP decides to trust the IdP's *statement* about who the user is. This is where SAML, OIDC, or OAuth come in.

> **The IdP authenticates the user. The Service Provider authenticates the IdP's statement about the user** — it does not re-run the original authentication.

### End-to-end flow

```mermaid
sequenceDiagram
    participant U as User
    participant SP as Service Provider (App)
    participant IDP as Identity Provider

    U->>SP: (1) Request application
    SP->>IDP: (2) Redirect: not authenticated
    U->>IDP: (3) Authenticate (password / MFA / etc.)
    IDP-->>U: (4) Signed identity artifact (SAML Assertion or OIDC ID Token)
    U->>SP: (5) Present artifact (via browser redirect)
    SP->>SP: (6) Verify signature, issuer, audience, expiry
    SP-->>U: (7) Set-Cookie: local session established
    Note over U,SP: Subsequent requests
    U->>SP: Cookie: SESSIONID=abc123
    SP-->>U: Authenticated response (IdP not contacted again)
```

Two points worth being precise about:

- **Verification is normally local, not a live call back to the IdP.** The SP validates the SAML assertion's XML signature or the OIDC ID Token's JWT signature using the IdP's published public key/metadata — it typically does *not* need to phone the IdP on every login. Exceptions exist for opaque OAuth access tokens, which do require introspection (§10), and for deployments that explicitly check token/session revocation status.
- **After the initial exchange, the application usually falls back to its own local session (cookie) for subsequent requests** — the SAML assertion or ID Token is not resent on every page load. This is the traditional web-app pattern. **Single-page apps and mobile apps often differ:** since they frequently skip server-side sessions, they instead keep re-presenting the (short-lived) OAuth access token as a `Bearer` header on every API call, with the token refreshed in the background via the refresh token.

### Why SSO reduces risk (and what it doesn't fix)

```mermaid
flowchart LR
    subgraph Without SSO
        User1[User] --> App1
        User1 --> App2
        User1 --> App3
        App1 --> PW1[Password 1]
        App2 --> PW2[Password 2]
        App3 --> PW3[Password 3]
    end
```

Fewer passwords means a smaller attack surface for credential-based attacks and simpler offboarding (disable one IdP account instead of N app accounts). It does **not** eliminate the bearer-token theft risk discussed in §9 — a stolen SSO session cookie or ID token is just as usable by an attacker as a stolen password-based session would be, so the same mitigations (short lifetimes, `HttpOnly`/`Secure` cookies, PoP where feasible) still apply.

---

## 13. SAML 2.0

**Security Assertion Markup Language** — an XML-based standard predating OIDC, still dominant in enterprise, government, and education environments (often via ADFS, Okta, or Entra ID).

**Roles:** the **Identity Provider (IdP)** authenticates users; the **Service Provider (SP)** is the application relying on that authentication.

```mermaid
sequenceDiagram
    participant U as User
    participant SP as Service Provider
    participant IDP as Identity Provider
    U->>SP: Access application
    SP->>IDP: Authentication request
    IDP->>U: Login page
    U->>IDP: Authenticate
    IDP-->>U: Signed SAML Assertion
    U->>SP: Forward assertion
    SP->>SP: Validate XML signature, issuer, audience, expiry
    SP-->>U: Access granted (local session created)
```

**Assertion example (simplified):**
```xml
<Assertion>
  <Subject>Mehrab</Subject>
  <AttributeStatement>
    <Attribute Name="Role">Admin</Attribute>
  </AttributeStatement>
</Assertion>
```
The assertion is XML-digitally-signed (XML-DSig). As with JWTs, the signature proves authenticity and integrity, not rightful possession (§9) — SAML deployments mitigate this primarily through short assertion validity windows (often 1–5 minutes) and one-time-use enforcement at the SP.

**SAML vs. OIDC, practically:** SAML is XML/browser-redirect-based and has no native concept of API access tokens — it's built for web SSO, not for authorizing calls to a REST API. OIDC/OAuth is JSON-based, works naturally for both web SSO and API authorization, and is the more common choice for new systems; SAML persists mainly due to enterprise IdP entrenchment.

---

## 14. Keycloak: A Reference IdP Implementation

Keycloak is an open-source Identity and Access Management (IAM) platform that implements the protocols above (OAuth 2.0, OIDC, SAML 2.0) plus LDAP/Active Directory federation, so individual applications don't each need to build authentication, MFA, password reset, and user management from scratch.

```mermaid
flowchart TB
    KC["Keycloak<br/>(IdP, user DB, MFA, SSO, OAuth2/OIDC/SAML)"]
    KC --> AppA[Web App]
    KC --> AppB[Mobile App]
    KC --> AppC[API]
    KC --> LDAP[LDAP / Active Directory]
    KC --> Social[Google / GitHub / Azure AD]
```

### Core concepts

- **Realm** — an isolated security domain: separate users, roles, clients, and authentication policy per realm (e.g., a `company` realm and a separate `university` realm on the same Keycloak instance).
- **Client** — an application registered to trust the realm (a web app, mobile app, or API).
- **User** — an identity managed within a realm (credentials, attributes, MFA config).
- **Roles / Groups** — authorization primitives; roles are permissions, groups are collections of users that can inherit roles for easier administration.

### Protocol fit

| Protocol | Keycloak's role | Typical client |
|---|---|---|
| OIDC | OpenID Provider (OP) | Modern web/mobile apps needing login |
| OAuth 2.0 | Authorization Server | APIs, service-to-service calls |
| SAML 2.0 | Identity Provider (IdP) | Enterprise apps (e.g., internal portals expecting SAML) |
| LDAP/AD federation | Delegates authentication to an existing directory | Organizations with an existing user directory |

A typical deployment sits behind a reverse proxy, serves multiple client applications, and persists state (users, sessions, clients) to a relational database — PostgreSQL is the most common production choice, though it isn't a hard requirement.

---

## 15. Comparison Tables

### Authentication method overview

| Method | Core mechanism | State | Typical use |
|---|---|---|---|
| Basic Auth | Base64 credentials, resent every request | Stateless | Internal APIs, testing |
| Digest Auth | Challenge-response hash | Stateless | Legacy systems |
| API Key | Static application secret | Usually stateless | Public/partner APIs |
| Session | Server-held state, client holds a reference | Stateful | Traditional web apps |
| Bearer / Opaque Token | Client-held credential, server or DB lookup | Stateless (client) / needs lookup (server) | REST APIs |
| JWT | Self-contained signed token | Stateless | Distributed systems, microservices |
| OAuth 2.0 | Delegated, scoped authorization | Depends on token type | Third-party access, M2M |
| OIDC | Authentication layer over OAuth | Depends on token type | Modern federated login |
| SAML | XML-based federated authentication | Stateless (assertion) + local session at SP | Enterprise SSO |

### Identity ecosystem, top to bottom

```mermaid
flowchart TB
    Identity --> Authentication
    Identity --> Authorization
    Authentication --> Password
    Authentication --> Session
    Authentication --> Tokens
    Tokens --> Bearer
    Tokens --> JWT
    Authorization --> OAuth2["OAuth 2.0"]
    Identity --> OIDCNode["OIDC (authN over OAuth)"]
    OIDCNode --> IDToken[ID Token]
    OAuth2 --> AccessToken[Access Token]
    SSO --> SAML
    SSO --> OIDCNode
```

---

## 16. Security Notes and Common Pitfalls

A consolidated checklist, since these recur across almost every scheme above:

1. **TLS is non-negotiable** for Basic Auth, Bearer tokens, session cookies, and OAuth/OIDC flows — without it, credentials and tokens are exposed in transit regardless of how they're encoded or signed.
2. **A signature proves authenticity, not possession.** Any bearer artifact (session cookie, opaque token, JWT, SAML assertion, OIDC ID token) is usable by whoever holds it, full stop, until it expires or is explicitly revoked. Design around this assumption rather than around the hope that theft won't happen.
3. **Prefer short-lived access credentials plus a longer-lived, rotated renewal credential** (access + refresh token) over one long-lived credential — this is the single highest-leverage mitigation against token theft.
4. **Don't trust the token's own header for validation logic.** A JWT verifier must pin the expected algorithm and key server-side; trusting the `alg` field from the token itself enables algorithm-confusion attacks.
5. **Never put secrets or unnecessary PII in a JWT payload** — it's encoded, not encrypted, and readable by anyone holding the token.
6. **API keys authenticate applications, not users** — don't rely on an API key as a proxy for user-level authorization without an explicit mapping.
7. **PKCE should be used for all authorization-code flows**, not only public clients, per current OAuth best practice (OAuth 2.1 direction).
8. **Revocation is the recurring weak point of stateless tokens.** If your system needs instant revocation (e.g., for compromised-account response), either keep tokens short-lived, maintain a deny-list for the rare high-severity case, or lean on opaque tokens + introspection where the extra round trip is acceptable.