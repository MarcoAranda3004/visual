import streamlit as st
import pandas as pd
import os

def mostrar_admin():
    st.title("🔐 Panel Administrativo")

    # Inicializar la variable de sesión para el estado del login
    if "autenticado" not in st.session_state:
        st.session_state["autenticado"] = False

    # Formulario de Login Simple
    if not st.session_state["autenticado"]:
        st.subheader("🔑 Acceso Restringido")
        
        with st.form("formulario_login"):
            usuario = st.text_input("Usuario admin:")
            contrasena = st.text_input("Contraseña:", type="password")
            boton_ingresar = st.form_submit_button("Ingresar")
            
            if boton_ingresar:
                # Cambia aquí tu usuario y clave cuando desees
                if usuario == "admin" and contrasena == "agro2@26":
                    st.session_state["autenticado"] = True
                    st.success("¡Acceso concedido!")
                    st.rerun()
                else:
                    st.error("Credenciales incorrectas.")
    else:
        # Panel de Control Privado (Solo visible tras loguearse)
        st.subheader("🛠️ Panel de Actualización de Datos")
        
        if st.button("Cerrar Sesión"):
            st.session_state["autenticado"] = False
            st.rerun()
            
        st.markdown("---")
        st.markdown("#### Actualizar Bases de Datos CSV")
        st.info("Espacio listo para implementar la carga de nuevos archivos (.csv) para mejorar las predicciones futuras.")