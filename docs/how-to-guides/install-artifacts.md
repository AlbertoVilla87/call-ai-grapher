# How to install Azure Devops Artifacts

!!! info "About This Section"
    Install libraries from Azure Devops Artifacts

## Steps

### 1. Create Access token in Azure Devops

Create a personal access token in Azure DevOps with "Packaging (Read)" permissions. Temporary store the token somewhere, because you will need it later.

### 2. Install the package `keyring`

```
pip install keyring
```

!!! warning
      Deactivate any virtual environment.
      
### 3. Configure authentication for the Azure DevOps artifacts feed

```
echo "your-personal-access-token" | keyring set pkgs.dev.azure.com artifacts
```

!!! warning
      You'll need to execute this again every time your token expires.

### 4. Configure Poetry to prevent SSL issues

Guide Poetry to your firewall root certificate.

```
poetry config certificates.devops.cert "path/cert.pem"
```

!!! info
      The name `devops` will be referenced in the projects, but you are free to choose another name.

!!! info      
      The ca-bundle.crt file is a bundle of trusted certificate authorities (CAs) used for verifying the authenticity of SSL/TLS certificates during secure connections. The specific location of the ca-bundle.crt file can vary depending on the operating system. Here are some common locations:
      
      On Debian/Ubuntu-based systems:

      ```
      /etc/ssl/certs/ca-certificates.crt
      ```

      On Red Hat/Fedora-based systems:

      ```
      /etc/pki/tls/certs/ca-bundle.crt
      ```

      On macOS:

      ```
      /etc/ssl/cert.pem
      ```

      On Windows:

      You can download it from trusted sources such as the curl project: https://curl.haxx.se/docs/caextract.html
      Alternatively, you may find it in your system configuration (location may vary depending on the Windows version):

      ```
      C:\Program Files\Common Files\SSL\cacert.pem
      ```

### 5. Add the Azure DevOps artifacts feed as an additional package source

```
poetry source add --priority=supplemental repo_name 
```

### 6. Add any package available in the artifacts feed

```
poetry add --source devops dociq-information-extractor==0.0.17
```
